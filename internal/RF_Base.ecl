IMPORT $.^ AS LT;
IMPORT LT.LT_Types as Types;
IMPORT ML_Core as ML;
IMPORT ML.Types AS CTypes;
IMPORT std.system.Thorlib;
IMPORT LT.ndArray;

GenField := Types.GenField;
ModelStats := Types.ModelStats;
t_Work_Item := CTypes.t_Work_Item;
t_Count := CTypes.t_Count;
t_RecordId := CTypes.t_RecordID;
t_FieldNumber := CTypes.t_FieldNumber;
t_TreeId := t_FieldNumber;
Layout_Model := CTypes.Layout_Model;
wiInfo := Types.wiInfo;
TreeNodeDat := Types.TreeNodeDat;
NumericField := CTypes.NumericField;
DiscreteField := CTypes.DiscreteField;
Layout_Model2 := Types.Layout_Model2;
rfModInd1 := Types.rfModInd1;
rfModNodes3 := Types.rfModNodes3;

/**
  * Base Module for Random Forest algorithms.  Modules for RF Classification or Regression
  * are based on this one.
  * It provides the attributes to set up the forest as well as s
  */
EXPORT RF_Base(DATASET(GenField) X_in,
              DATASET(GenField) Y_In,
              UNSIGNED numTrees=100,
              UNSIGNED featuresPerNodeIn=0,
              UNSIGNED maxDepth=255) := MODULE, VIRTUAL
  SHARED autoBin := TRUE;
  SHARED autobinSize := 10;
  SHARED autobinSizeScald := autobinSize * 4294967295;

  SHARED X0 := DISTRIBUTE(X_in, HASH32(wi, id));
  SHARED Y := DISTRIBUTE(Y_in, HASH32(wi, id));
  SHARED X := SORT(X0, wi, id, number, LOCAL);
  SHARED Rand01 := RANDOM()/4294967295; // Random number between zero and one.

  // P log P calculation for entropy.  Note that Shannon entropy uses log base 2 so the division by LOG(2) is
  // to convert the base from 10 to 2.
  SHARED P_Log_P(REAL P) := IF(P=1, 0, -P* LN(P) / LN(2));

  SHARED empty_model := DATASET([], Layout_Model2);

  // Calculate work-item metadata
  Y_S := SORT(Y, wi);  // Sort Y by work-item
  // Each work-item needs its own metadata (i.e. numSamples, numFeatures, .  Construct that here.
  wiMeta0 := TABLE(X, {wi, numSamples := MAX(GROUP, id), numFeatures := MAX(GROUP, number), featuresPerNode := 0}, wi);
  wiInfo makeMeta(wiMeta0 lr) := TRANSFORM
    // If featuresPerNode was passed in as zero (default), use the square root of the number of features,
    // which is a good rule of thumb.  In general, with multiple work-items of different sizes, it is best
    // to default featuresPerNode.
    fpt0 := IF(featuresPerNodeIn > 0, featuresPerNodeIn, TRUNCATE(SQRT(lr.numFeatures)));
    // In no case, let features per tree be greater than the number of features.
    SELF.featuresPerNode := MIN(fpt0, lr.numFeatures);
    SELF := lr;
  END;
  SHARED wiMeta := PROJECT(wiMeta0, makeMeta(LEFT));

  // Data structure to hold the sample indexes (i.e Bootstrap Sample) for each treeId
  SHARED sampleIndx := RECORD
    t_TreeID treeId;
    t_RecordId id;     // Id within this tree
    t_RecordId origId; // The id of this sample in the original X,Y
  END;

  // treeSampleIndx has  <samples> sample indexes for each tree, sorted by tree.  This represents
  // the "Bootstrap Sample" for each tree using sampling with replacement.
  // It is used during tree initialization, and is also needed for analytics / validation so that
  // "out-of-bag" (OOB) samples can be created.  Use all cluster nodes to build the index, and
  // leave it distributed by tree-id.

  // Note: The approach is somewhat strange, but done for distributed performance.
  // Start from the samples.  Generate enough samples so that there are enough for the work-item
  // with the most samples.  We'll use truncations of these samples for the same treeId across work-items.
  // So we only need to create the sampling index once per treeId.
  SHARED maxSampleSize := MAX(wiMeta, numSamples); // maximum samples for any work-item
  SHARED maxfeaturesPerNode := MAX(wiMeta, featuresPerNode); // maximum features for any work-item
  dummy := DATASET([{0,0,0}], sampleIndx);
  // Create one dummy sample per tree
  treeDummy := NORMALIZE(dummy, numTrees, TRANSFORM(sampleIndx, SELF.treeId := COUNTER, SELF := []));
  // Distribute by treeId to create the samples in parallel
  treeDummyD := DISTRIBUTE(treeDummy, treeId);
  // Now generate samples for each treeId in parallel
  SHARED treeSampleIndx :=NORMALIZE(treeDummyD, maxSampleSize, TRANSFORM(sampleIndx, SELF.origId := (RANDOM()%maxSampleSize) + 1, SELF.id := COUNTER, SELF := LEFT));

  // Function to randomly select features to use for each level of the tree building.
  // Each node is assigned a random subset of the features.
  SHARED DATASET(TreeNodeDat) SelectVarsForNodes(DATASET(TreeNodeDat) nodeDat) := FUNCTION
    // At this point, nodeDat should have one instance per id per node per tree per wi, distributed by (wi, treeId)
    // Nodes should be sorted by (at least) wi, treeId, nodeId at this point.
    // We are trying to choose featuresPerNode features out of the full set of features for each tree node
    // First, extract the set of treeNodes
    nodes := DEDUP(nodeDat, wi, treeId, nodeId, LOCAL);  // Now we have one record per node
    // Now, extend the the tree data.  Add a random number field and create <features> records for each tree.
    xTreeNodeDat := RECORD(TreeNodeDat)
      UNSIGNED numFeatures;
      UNSIGNED featuresPerNode;
      UNSIGNED rnd;
    END;
    // Note that each work-item may have a different value for numFeatures and featuresPerNode
    xTreeNodeDat makeXNodes(treeNodeDat l, wiInfo r) := TRANSFORM
      SELF.numFeatures := r.numFeatures;
      SELF.featuresPerNode := r.featuresPerNode;
      SELF := l;
      SELF := [];
    END;
    xNodes := JOIN(nodes, wiMeta, LEFT.wi = RIGHT.wi, makeXNodes(LEFT, RIGHT), LOOKUP, FEW);
    xTreeNodeDat getFeatures(xTreeNodeDat l, UNSIGNED c) := TRANSFORM
      // Choose twice as many as we need, so that when we remove duplicates, we will (almost always)
      // have at least the right number.  This is more efficient than enumerating all and picking <featuresPerNode>
      // from that set because numFeatures >> featuresPerNode.  We will occasionally get a tree that
      // has less than <featuresPerNode> variables, but that should only add to the diversity.
      nf := l.numFeatures;
      SELF.number := (RANDOM()%nf) + 1;
      SELF.rnd := RANDOM();
      SELF := l;
    END;
    // Create twice as many features as we need, so that when we remove duplicates, we almost always
    // have at least as many as we need.
    nodeVars0 := NORMALIZE(xNodes, LEFT.featuresPerNode * 2, getFeatures(LEFT, COUNTER));
    nodeVars1 :=  GROUP(nodeVars0, wi, treeId, nodeId, LOCAL);
    nodeVars2 := SORT(nodeVars1, wi, treeId, nodeId, number); // Note: implicitly local because of GROUP
    // Get rid of any duplicate features (we sampled with replacement so may be dupes)
    nodeVars3 := DEDUP(nodeVars2, wi, treeId, nodeId, number);
    // Now we have up to <featuresPerNode> * 2 unique features per node.  We need to whittle it down to
    // no more than <featuresPerNode>.
    nodeVars4 := SORT(nodeVars3, wi, treeId, nodeId, rnd); // Mix up the features
    // Filter out the excess vars and transform back to TreeNodeDat.  Set id (not yet used) just as an excuse
    // to check the count and skip if needed.
    nodeVars := UNGROUP(PROJECT(nodeVars4, TRANSFORM(TreeNodeDat,
                    SELF.id := IF(COUNTER <= LEFT.featuresPerNode, 0, SKIP),
                    SELF := LEFT)));
    // At this point, we have <featuresPerNode> records for almost every node.  Occasionally one will have less
    // (but at least 1).
    // Now join with original nodeDat (one rec per tree node per id) to create one rec per tree node per id per
    // selected feature.
    nodeVarDat := JOIN(nodeDat, nodeVars, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId AND LEFT.nodeId = RIGHT.nodeId,
                          TRANSFORM(TreeNodeDat, SELF.number := RIGHT.number, SELF := LEFT), LOCAL);
    RETURN nodeVarDat;
  END;

  // Sample with replacement <samples> items from X,Y for each tree
  SHARED DATASET(TreeNodeDat) GetBootstrapForTree(DATASET(TreeNodeDat) trees) := FUNCTION
    // At this point, trees contains one record per tree for each wi
    // Use the bootstrap (treeSampleIndxs) built at the module level

    // Note: At this point, trees and treeSampleIndx are both sorted and distributed by
    // treeId
    // We need to add the sample size from the wi to the dataset in order to filter appropriately
    xtv := RECORD(TreeNodeDat)
      t_RecordId numSamples;
    END;
    xTrees := JOIN(trees, wiMeta, LEFT.wi = RIGHT.wi, TRANSFORM(xtv, SELF.numSamples := RIGHT.numSamples,
                          SELF := LEFT), LOOKUP, FEW);
    // Expand the trees to include the sample index for each tree.
    // Size is now <numTrees>  * <maxSamples> per wi
    // Note: this is a many to many join.
    treeDat0 := JOIN(xTrees, treeSampleIndx, LEFT.treeId = RIGHT.treeId,
                        TRANSFORM(xtv, SELF.origId := RIGHT.origId, SELF.id := RIGHT.id, SELF := LEFT),
                        MANY, LOOKUP);
    // Filter treeDat0 to remove any samples with origId > numSamples for that wi.
    // The number of samples will not (in all cases) be = the desired sample size, but shouldn't create any bias.
    // This was our only need for numSamples, so we project back to TreeNodeDat format
    treeDat1 := PROJECT(treeDat0(origId <= numSamples), TreeNodeDat);
    // Now redistribute by wi and <origId> to match the Y data
    treeDat1D := DISTRIBUTE(treeDat1, HASH32(wi, origId));

    // Now get the  corresponding Y (dependent) value
    // While we're at it, assign the data to the root (i.e. nodeId = 1, level = 1)
    treeDat := JOIN(treeDat1D, Y, LEFT.wi = RIGHT.wi AND LEFT.origId=RIGHT.id,
                        TRANSFORM(TreeNodeDat, SELF.depend := RIGHT.value, SELF.nodeId := 1, SELF.level := 1, SELF := LEFT),
                        LOCAL);
    // At this point, we have one instance per tree  per sample, for each work-item, and each instance
    // includes the Y values for the selected indexes (i.e. depend)
    // TreeDat is distributed by work-item and sample id.
    RETURN treeDat;
  END;

  // Create the set of tree definitions -- One single node per tree (the root), with all tree samples associated with that root.
  SHARED DATASET(TreeNodeDat) InitTrees := FUNCTION
    // Create an empty tree data instance per work-item
    dummyTrees := PROJECT(wiMeta, TRANSFORM(TreeNodeDat, SELF.wi := LEFT.wi, SELF := []));
    // Use that to create "numTrees" dummy trees -- a dummy (empty) forest per wi
    trees := NORMALIZE(dummyTrees, numTrees, TRANSFORM(TreeNodeDat, SELF.treeId:=COUNTER, SELF.wi := LEFT.wi, SELF:=[]));
    // Distribute by wi and treeId
    treesD := DISTRIBUTE(trees, HASH32(wi, treeId));
    // Now, choose bootstrap sample of X,Y for each tree
    roots := GetBootstrapForTree(treesD);
    // At this point, each tree is fully populated with a single root node(i.e. 1).  All the data is associated with the root node.
    // Roots has each tree's bootstrap sample of the dependent variable (selected for the tree).
    // Roots is distributed by wi and origId (original sample index)
    RETURN roots;
  END;

  // Grow one layer of the forest.  Virtual method to be overlaid by specific (Classification or Regression)
  // module
  SHARED VIRTUAL DATASET(TreeNodeDat) GrowForestLevel(DATASET(TreeNodeDat) nodeDat, t_Count treeLevel) := FUNCTION
    return DATASET([], TreeNodeDat);
  END;

  // Grow a Classification Forest from a set of roots containing all the data points (X and Y) for each tree.
  SHARED DATASET(TreeNodeDat) GrowForest(DATASET(TreeNodeDat) roots) := FUNCTION
    // Localize all the data by wi and treeId
    rootsD := DISTRIBUTE(roots, HASH32(wi, treeId));
    // Grow the forest one level at a time.
    treeNodes  := LOOP(rootsD, LEFT.id > 0, (COUNTER <= maxDepth) AND EXISTS(ROWS(LEFT)) , GrowForestLevel(ROWS(LEFT), COUNTER));
    return SORT(treeNodes, wi, treeId, level, nodeId);
  END;

  // Generate all tree nodes for classification
  EXPORT DATASET(TreeNodeDat) GetNodes := FUNCTION
    // First create a set of tree roots, each
    // with a unique bootstrap sample out of X,Y
    roots := InitTrees;
    // We now have a single root node for each tree (level = 1, nodeId = 1).  All of the data is
    // associated with the root for each tree.
    // Now we want to grow each tree by adding nodes, and moving the data
    // points to lower and lower nodes at each split.
    // When we are done, all of the data will be gone and all that will remain
    // is the skeleton of the decision tree with splits and leaf nodes.
    forestNodes := GrowForest(roots);
    // We now just have the structure of the decision trees remaining.  All data
    // is now summarized by the trees' structure into leaf nodes.
    RETURN forestNodes;
  END;

  /**
    * Extract the set of tree nodes from a model
    *
    */
  EXPORT DATASET(TreeNodeDat) Model2Nodes(DATASET(Layout_Model2) mod) := FUNCTION
    // Extract nodes from model as NumericField dataset
    nfNodes := ndArray.ToNumericField(mod, [rfModInd1.nodes]);
    // Distribute by wi and id for distributed processing
    nfNodesD := DISTRIBUTE(nfNodes, HASH32(wi, id));
    nfNodesG := GROUP(nfNodesD, wi, id, LOCAL);
    nfNodesS := SORT(nfNodesG, wi, id, number);
    TreeNodeDat makeNodes(NumericField rec, DATASET(NumericField) recs) := TRANSFORM
      SELF.wi := rec.wi;
      SELF.treeId := recs[rfModNodes3.treeId].value;
      SELF.level := recs[rfModNodes3.level].value;
      SELF.nodeId := recs[rfModNodes3.nodeId].value;
      SELF.parentId := recs[rfModNodes3.parentId].value;
      SELF.isLeft := recs[rfModNodes3.isLeft].value = 1;
      SELF.number := recs[rfModNodes3.number].value;
      SELF.value := recs[rfModNodes3.value].value;
      SELF.isOrdinal := recs[rfModNodes3.isOrdinal].value = 1;
      SELF.depend := recs[rfModNodes3.depend].value;
      SELF.support := recs[rfModNodes3.support].value;
      SELF := [];
    END;
    // Rollup individual fields into TreeNodeDat records.
    nodes := ROLLUP(nfNodesS, GROUP, makeNodes(LEFT, ROWS(LEFT)));
    // Distribute by wi and TreeId
    //nodes := DISTRIBUTE(nodes0, HASH32(wi, treeId));
    RETURN nodes;
  END;
  /**
    * Extract the set of sample indexes (i.e. bootstrap samples for each tree)
    * from a model
    *
    */
  EXPORT Model2Samples(DATASET(Layout_Model2) mod) := FUNCTION
    nfSamples := ndArray.ToNumericField(mod, [rfModInd1.samples]);
    samples := PROJECT(nfSamples, TRANSFORM(sampleIndx, SELF.treeId := LEFT.id,
                                            SELF.id := LEFT.number,
                                            SELF.origId := LEFT.value));
    return samples;
  END;
  /**
    * Convert the set of nodes describing the forest to a Model Format
    *
    */
  EXPORT DATASET(Layout_Model2) Nodes2Model(DATASET(TreeNodeDat) nodes) := FUNCTION
    NumericField makeMod({TreeNodeDat, UNSIGNED recordId} d, UNSIGNED c) := TRANSFORM
      SELF.wi := d.wi;
      indx1 := CHOOSE(c, rfModNodes3.treeId, rfModNodes3.level, rfModNodes3.nodeId,
                         rfModNodes3.parentId, rfModNodes3.isLeft, rfModNodes3.number,
                         rfModNodes3.value, rfModNodes3.isOrdinal,
                         rfModNodes3.depend, rfModNodes3.support);
      SELF.value := CHOOSE(c, d.treeId, d.level, d.nodeId, d.parentId,
                            (UNSIGNED)d.isLeft, d.number, d.value, (UNSIGNED)d.isOrdinal,
                            d.depend, d.support);
      SELF.number := indx1;
      SELF.id := d.recordId;
    END;
    // Add a record id to nodes
    nodesExt := PROJECT(nodes, TRANSFORM({TreeNodeDat, UNSIGNED recordId}, SELF.recordId := COUNTER, SELF := LEFT));
    // Make into a NumericField dataset
    nfMod := NORMALIZE(nodesExt, 10, makeMod(LEFT, COUNTER));
    // Insert at position [modInd.nodes] in the ndArray
    mod := ndArray.FromNumericField(nfMod, [rfModInd1.nodes]);
    RETURN mod;
  END;
  /**
    * Convert the set of tree sample indexes to a Model Format
    *
    */
  SHARED Indexes2Model := FUNCTION
    nfIndexes := PROJECT(treeSampleIndx, TRANSFORM(NumericField,
                                                    SELF.wi := 0, // Not used
                                                    SELF.id := LEFT.treeId,
                                                    SELF.number := LEFT.id,
                                                    SELF.value := LEFT.origId));
    indexes := ndArray.FromNumericField(nfIndexes, [rfModInd1.samples]);
    return indexes;
  END;
  /**
    * Get forest model
    *
    * RF uses the Layout_Model2 format, which is implemented as an N-Dimensional
    * numeric array (i.e. ndArray.NumericArray).
    *
    * See LT_Types for the format of the model
    *
    */
  EXPORT DATASET(Layout_Model2) GetModel := FUNCTION
    nodes := GetNodes;
    mod1 := Nodes2Model(nodes);
    mod2 := Indexes2Model;
    mod := mod1 + mod2;
    RETURN mod;
  END;

  // ModelStats
  EXPORT GetModelStats(DATASET(Layout_Model2) mod) := FUNCTION
    nodes := Model2Nodes(mod);
    treeStats := TABLE(nodes, {wi, treeId, nodeCount := COUNT(GROUP), depth := MAX(GROUP, level),
                        totSupport := SUM(GROUP, support)}, wi, treeId);
    topStats := TABLE(treeStats, {wi, treeCount := COUNT(GROUP),
                        minTreeDepth := MIN(GROUP, depth), maxTreeDepth := MAX(GROUP, depth),
                        avgTreeDepth := AVE(GROUP, depth),
                        minTreeNodes := MIN(GROUP, nodeCount), maxTreeNodes := MAX(GROUP, nodeCount),
                        avgTreeNodes := AVE(GROUP, nodeCount),
                        maxNodesPerTree := MAX(GROUP, nodeCount), totalNodes := SUM(GROUP, nodeCount),
                        minSupport := MIN(GROUP, totSupport), maxSupport := MAX(GROUP, totSupport),
                        avgSupport := AVE(GROUP, totSupport)}, wi);
    RETURN PROJECT(topStats, ModelStats);
  END;
END; // RF_Base
