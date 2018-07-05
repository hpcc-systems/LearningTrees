/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2017 HPCC Systems®.  All rights reserved.
############################################################################## */
IMPORT $.^ AS LT;
IMPORT LT.Internal AS int;
IMPORT LT.LT_Types as Types;
IMPORT ML_Core as ML;
IMPORT ML.Types AS CTypes;
IMPORT std.system.Thorlib;
IMPORT ML_Core.ModelOps2;

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
Layout_Model2 := CTypes.Layout_Model2;
FeatureImportanceRec := Types.FeatureImportanceRec;
nfNull := DATASET([], NumericField);

/**
  * Base Module for Random Forest algorithms.  Modules for RF Classification or Regression
  * are based on this one.
  * The random forest algorithm used is based on Brieman 2001 with extensions.
  * It provides the attributes to set up the forest as well as storing of the resulting forest
  * in a common model format.  It also provides various analytic methods.
  *
  * Theory of Operation
  *
  * The heart of the random forest algorithm is the building of a forest model (multiple
  * diverse decision trees) by segmenting the training data through binary splits that
  * increase the uniformity of the dependent variable as the tree is developed.
  * In random forest, the trees are developed until the data after the final split is
  * totally pure (i.e. all members have the same value of the dependent variable), or
  * a maximum tree depth is encountered (configurable -- see maxDepth parameter).  If a
  * maximum depth is encountered with an impure group on either or both sides, the remaining
  * points are aggregated through an appropriate means (implementation dependent).
  * When a pure group is obtained (either through purity or aggregation), the data points
  * in that group are summarized by a "leaf node".  Each tree is therefore composed of
  * split-nodes (i.e. branches) and leaf-nodes.
  * The resulting forest model can then be used to perform predictions of the dependent
  * variable given hitherto unseen samples of the independent variable.  In the case of
  * a Regression Forest, the unobserved dependent variable is predicted as a continuous
  * number.  In the case of a Classification Forest, the dependent variable takes the form
  * of a discrete class identifier, and the prediction method is called "Classify".
  * This implementation allows the independent variables to be continuous or discrete,
  * and supports both ordered variables (e.g. quantitative -- real-numbers, integers) and
  * categorical (aka "nominal") variables, which can represent qualitative info such as
  * type, color, state, which are encoded with integer values. These different types of
  * independent variables are encoded using the GenField dataset layout.
  * Once a forest model has been created, new values are predicted by running each
  * datapoint down the tree for each tree in the forest, arriving at a single leaf for
  * each datapoint in each tree.  The final prediction is arrived at by aggregating
  * the leaf value of all the trees in the forest for that point.  Depending on whether
  * we are classifying or predicting a continous value (i.e. regression), a different
  * aggregation method is used.  For classification, a voting method is used.  The most
  * commonly predicted class is the final result.  For regression, the mean of all the
  * trees' predictions is returned as the final result.
  * Each tree can be thought of as an 'overfit' model since they will fit to not only
  * the 'signal' in the data, but also the 'noise'.  The aggregation across the forest
  * averages out most of the noise (due to the diversity of the trees).  This is a
  * key strength of the random forest algorithm as it makes it robust against over-fitting.
  * The diversity of the forest is achieved in a number of ways:
  * 1) Each tree is built using a bootstrap of the original data.  That is, for each
  *    tree, the data is sampled with replacement, resulting in a different set of
  *    points used to create the tree.  On average, aroung 30% of the datapoints will
  *    be missing from each tree due to replication of other points.
  * 2) Each split node is created based on a random subset of features (the best split
  *    for each subset is used).  This randomly eliminates some possible branches
  *    for each split and causes each tree to grow in a unique fashion.
  * 3) Unless auto-binning is disabled, for variables with many values (e.g. continuous)
  *    the possible split points are randomly sampled so that only a smaller set of
  *    points are considered for each split.  This technique provides better performance
  *    while increasing the diversity of the trees in the forest, and is sometimes referred-
  *    to as "extremely randomized foests".  This is the default mode of operation as it
  *    seems to provide strong benefits without significant down-side.
  * This base module provides the common methods for building and using the forest model,
  * while the derived modules (e.g. ClassificationForest and RegressionForest) overlay
  * virtual attributes that perform work specific to the type of forest.
  * The basic steps in building the tree are:
  * - Organize the input data:
  *   - Identify the work-items and number of training records and features
  *   - Ensure that the record ids are sequential
  *   - Create a map of each of the feature numbers for each work-item
  * - Generate random bootstrap samples for each tree
  * - Create numTree tree root nodes
  * - Replicate the training data bootstraps to each tree root
  * - Build the trees, one layer at a time by:
  *   - Randomly choose the featuresPerNode features to evaluate for a given level
  *   - Evaluate all possible splits on those features to find the feature and split-point
  *     that results in the most pure pair of sub-groups.  The criteria for evaluating
  *     splits is different for the different forest types.
  *   - Choose the best split, and create a split-node (branch) based on that split.
  *   - Create a left and right child node at the next level under the split-node.
  *   - Reallocate the data points from the node to the left or right child nodes.
  *   - If any group is pure (i.e. same dependent value for all members) or we've reached
  *     maxDepth, replace all the datapoints in that node with a single leaf node.
  * - At the end of the process, all the datapoints will have been absorbed by leaf nodes,
  *   and the resulting trees will contain only split-nodes and leaf-nodes.
  * - Note that a single common data structure (see LT_Types.TreeNodeData) is used to
  *   represent:
  *   - Data-points that need to be processed for a node (only during tree growing)
  *   - A split-node
  *   - A leaf-node
  * - The resulting tree-nodes representing the forest are converted to a Forest Model
  *   for later use in prediction or analysis.
  */
EXPORT RF_Base(DATASET(GenField) X_in=DATASET([], GenField),
              DATASET(GenField) Y_In=DATASET([], GenField),
              UNSIGNED numTrees=100,
              UNSIGNED featuresPerNodeIn=0,
              UNSIGNED maxDepth=255,
              DATASET(NumericField) observWeights=nfNull) := MODULE
  SHARED haveObsWeights := EXISTS(observWeights);
  // Resequence ids to go from 1-numRecords.
  SHARED resequenceByWi(DATASET(GenField) dat) := FUNCTION
    // Dat should be evenly distributed at this point
    xGen := RECORD(GenField)
      t_RecordId newId := 0;
    END;
    xDat := PROJECT(dat, TRANSFORM(xGen, SELF := LEFT));
    xGen setNewIds(xGen l, xGen r) := TRANSFORM
      newId := IF(l.wi != r.wi, 1, IF(r.id = l.id, l.newId, l.newId + 1));
      SELF.newId := newId;
      SELF := r;
    END;
    xDatS := SORT(xDat, wi, id);
    xDat2 := ITERATE(xDatS, setNewIds(LEFT, RIGHT));
    outDat0 := PROJECT(xDat2, TRANSFORM(GenField, SELF.id := LEFT.newId, SELF := LEFT));
    outDat := DISTRIBUTE(outDat0, HASH32(wi, id));
    return outDat;
  END;
  SHARED autoBin := TRUE;
  SHARED autobinSize := 10;
  SHARED maxU4 := 4294967295; // Maximum value for an Unsigned 4
  SHARED maxR8 := 1.797693e+308; // Maximum value for a REAL8
  SHARED autobinSizeScald := autobinSize * maxU4; // Scaled auto-bin size for efficiency
  SHARED XD := DISTRIBUTE(X_in, HASH32(wi, id));
  SHARED YD := DISTRIBUTE(Y_in, HASH32(wi, id));
  SHARED XSD := SORT(XD, wi, id, number, LOCAL);
  SHARED YSD := SORT(YD, wi, LOCAL);  // Sort Y by work-item
  SHARED Rand01 := RANDOM()/maxU4; // Random number between zero and one.

  // P log P calculation for entropy.  Note that Shannon entropy uses log base 2 so the division by LN(2) is
  // to convert the base from e to 2.
  SHARED P_Log_P(REAL P) := IF(P=1, 0, -P* LN(P) / LN(2));

  SHARED empty_model := DATASET([], Layout_Model2);
  SHARED empty_data := DATASET([], GenField);
  // Abbreviations for Model Index definitions
  SHARED FM := Types.Forest_Model;
  SHARED FM1 := FM.Ind1;
  SHARED FMN3 := FM.Ind3_nodes;
  // Calculate work-item metadata
  // Each work-item needs its own metadata (i.e. numSamples, numFeatures, .  Construct that here.
  SHARED wiSamples := TABLE(YSD, {wi, numSamples := COUNT(GROUP), maxId := MAX(GROUP, id)}, wi);
  idFeatures := TABLE(XSD, {wi, id, numFeats := COUNT(GROUP), maxFNum := MAX(GROUP, number)}, wi, id, LOCAL);
  SHARED wiFeatures := TABLE(idFeatures, {wi, numFeatures := MAX(GROUP, numFeats), maxNum := MAX(GROUP, maxFNum)}, wi);
  wiMeta0 := JOIN(wiSamples, wiFeatures, LEFT.wi = RIGHT.wi, TRANSFORM({wiSamples, UNSIGNED numFeatures},
                  SELF.numFeatures := RIGHT.numFeatures, SELF := LEFT));
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
  SHARED needsReseqTest := SUM(wiSamples, ABS(numSamples - maxId)) != 0;
  SHARED X := IF(needsReseqTest, resequenceByWi(XSD), XSD);
  SHARED Y := IF(needsReseqTest, resequenceByWi(YSD), YSD);
  // Create a map of feature number -> sequential feature number (1-numFeatures) for each wi.
  // Note: at this point, there has to be a record with id=1 for each work-item.
  // Overload the id field (which is not needed here) with a sequential id (1-numFeatures)
  // so that we can map between the two
  allFeatures := SORT(X(id=1), wi, number);
  allFeaturesG := GROUP(allFeatures, wi);
  featureMap0 := PROJECT(allFeaturesG, TRANSFORM(GenField, SELF.id := COUNTER, SELF := LEFT));
  SHARED featureMap := UNGROUP(featureMap0);
  SHARED needFeatureRenumbering := COUNT(featureMap) != SUM(wiFeatures, maxNum);
  SHARED allowNoProgress := IF(featuresPerNodeIn < MAX(featureMap, id), TRUE, FALSE);
                                   // If FALSE, tree will terminate when no progess can be made on any
                                   // feature.  For RF, should be TRUE since it may get a better choice
                                   // of features at the next level.  Set FALSE if featuresPerNode >= numFeatures,
                                   // since we will always be choosing from all features.
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
  // In the event that there is only one tree in the forest, the best result will
  // be gotten by using the full data (i.e. no sampling).
  treeSampleIndxSampled := NORMALIZE(treeDummyD, maxSampleSize, TRANSFORM(sampleIndx,
                            SELF.origId := (RANDOM()%maxSampleSize) + 1,
                            SELF.id := COUNTER,
                            SELF := LEFT));
  treeSampleIndxNonSampled := NORMALIZE(treeDummyD, maxSampleSize, TRANSFORM(sampleIndx,
                            SELF.origId := COUNTER,
                            SELF.id := COUNTER,
                            SELF := LEFT));
  SHARED treeSampleIndx := IF(numTrees > 1, treeSampleIndxSampled, treeSampleIndxNonSampled);
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
    nodeVars5 := UNGROUP(PROJECT(nodeVars4, TRANSFORM(TreeNodeDat,
                    SELF.id := IF(COUNTER <= LEFT.featuresPerNode, 0, SKIP),
                    SELF := LEFT)));
    // If the user provided features numbers that were not sequential, we need to map these feature numbers
    // (which are sequential) to the actual numbers the user provided.
    renumberFeatures(DATASET(TreeNodeDat) dat) := FUNCTION
      rnDat := JOIN(dat, featureMap, LEFT.wi = RIGHT.wi AND LEFT.number = RIGHT.id,
                      TRANSFORM(TreeNodeDat, SELF.number := RIGHT.number, SELF := LEFT), LOOKUP);
      RETURN rnDat;
    END;
    nodeVars := IF(needFeatureRenumbering, renumberFeatures(nodeVars5), nodeVars5);
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
    treeDat2 := JOIN(treeDat1D, Y, LEFT.wi = RIGHT.wi AND LEFT.origId=RIGHT.id,
                        TRANSFORM(TreeNodeDat, SELF.depend := RIGHT.value, SELF.nodeId := 1, SELF.level := 1,
                        SELF.observWeight := 1.0, SELF := LEFT),
                        LOCAL);
    treeDat2w := JOIN(treeDat2, observWeights, LEFT.wi = RIGHT.wi AND LEFT.origId = RIGHT.id,
                      TRANSFORM(RECORDOF(LEFT), SELF.observWeight := IF(RIGHT.value > 0, RIGHT.value, 1.0),
                                  SELF := LEFT), LEFT OUTER, LOOKUP);
    treeDat := IF(haveObsWeights, treeDat2w, treeDat2);
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
    treeNodes  := LOOP(rootsD, LEFT.id > 0, EXISTS(ROWS(LEFT)(COUNTER <= maxDepth)) , GrowForestLevel(ROWS(LEFT), COUNTER));
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

  // Find the corresponding leaf node for each X sample given an expanded forest model (set of tree nodes)
  EXPORT DATASET(TreeNodeDat) GetLeafsForData(DATASET(TreeNodeDat) tNodes, DATASET(GenField) X) := FUNCTION
    // Distribute X by wi and id.
    x_D := SORT(DISTRIBUTE(X, HASH32(wi, id)), wi, id, LOCAL);
    x_ids := DEDUP(x_D, wi, id, LOCAL);
    // Extend each root for each ID in X
    // Leave the extended roots distributed by wi, id.
    roots := tNodes(level = 1);
    rootsExt := JOIN(x_ids, roots, LEFT.wi = RIGHT.wi, TRANSFORM(TreeNodeDat, SELF.id := LEFT.id, SELF := RIGHT),
                     MANY, LOOKUP);
    rootBranches := rootsExt(number != 0); // Roots are almost always branch (split) nodes.
    rootLeafs := rootsExt(number = 0); // Unusual but not impossible
    loopBody(DATASET(TreeNodeDat) levelBranches, UNSIGNED tLevel) := FUNCTION
      // At this point, we have one record per node, per id.
      // We extend each id down the tree one level at a time, picking the correct next nodes
      // for that id at each branch.
      // Next nodes are returned -- both leafs and branches.  The leafs are filtered out by the LOOP,
      // while the branches are sent on to the next round.
      // Ultimately, a leaf is returned for each id, which defines our final result.
      // Select the next nodes by combining the selected data field with each node
      // Note that:  1) we retain the id from the previous round, but the field number(number) is derived from the branch
      //             2) 'value' in the node is the value to split upon, while value in the data (X) is the value of that datapoint
      //             3) NodeIds at level n + 1 are deterministic.  The child nodes at the next level's nodeId is 2 * nodeId -1 for the
      //                left node, and 2 * nodeId for the right node.
      branchVals := JOIN(levelBranches, x_D, LEFT.wi = RIGHT.wi AND LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number,
                          TRANSFORM({TreeNodeDat, BOOLEAN branchLeft},
                                      SELF.branchLeft :=  ((LEFT.isOrdinal AND RIGHT.value <= LEFT.value) OR
                                                          ((NOT LEFT.isOrdinal) AND RIGHT.value = LEFT.value)),
                                      SELF.parentId := LEFT.nodeId, SELF := LEFT),
                          LOCAL);
      // Now, nextNode indicates the selected (left or right) nodeId at the next level for each branch
      // Now we use nextNode to select the node for the next round, for each instance
      nextLevelNodes := tNodes(level = tLevel + 1);
      nextLevelSelNodes := JOIN(branchVals, nextLevelNodes, LEFT.wi = RIGHT.wi AND
                              LEFT.treeId = RIGHT.treeId AND LEFT.nodeId = RIGHT.parentId AND
                              LEFT.branchLeft = RIGHT.isLeft,
                              TRANSFORM(TreeNodeDat, SELF.id := LEFT.id, SELF := RIGHT), LOOKUP);
      // Return the selected nodes at the next level.  These nodes may be leafs or branches.
      // Any leafs will be filtered out by the loop.  Any branches will go on to the next round.
      // When there are no more branches to process, we are done.  The selected leafs for each datapoint
      // is returned. All nodes are left distributed by wi, id.
      RETURN nextLevelSelNodes;
    END;
    // The loop will return the deepest leaf node associated with each sample.
    selectedLeafs0 := LOOP(rootBranches, LEFT.number>0, EXISTS(ROWS(LEFT)),
                          loopBody(ROWS(LEFT), COUNTER));
    selectedLeafs := selectedLeafs0 + rootLeafs;
    RETURN selectedLeafs;
  END; // GetLeafsForData
  /**
    * During RF training, we will occasionally hit a situation where all of the randomly selected
    * features for a level of the tree are constant for a node.  In this case, we are forced to insert
    * a dummy-split using a random feature from the selected subset in order to continue processing the
    * data.  We should see other useful features in subsequent rounds.  In this case, we set the splitVal
    * to MaxR8 (the maximum value for a REAL8 field) so that all data will take the left path.
    * This function removes such dummy splits and replaces that node with its child nodes, resulting
    * in a smaller tree which should be faster to process for prediction / classification of new points,
    * as well as any analytic operations.
    */
  SHARED DATASET(TreeNodeDat) CompressNodes(DATASET(TreeNodeDat) inNodes) := FUNCTION
    nodesS := SORT(DISTRIBUTE(inNodes, HASH32(wi, treeId)), wi, treeId, level, nodeId, LOCAL);
    // Assign a unique id to each node, independent of level.  We can re-use the id field
    // since it is not used at this point.  Note, these only need to be unique within a
    // tree, but using LOCAL PROJECT is efficient.  It will asssign a unique id to all
    // nodes across trees that are on the node.  Note that tree-nodes are distributed by
    // wi and treeId at this point.
    xNodes0 := PROJECT(nodesS, TRANSFORM(TreeNodeDat,
                                        SELF.id := COUNTER,
                                        SELF := LEFT), LOCAL);
    // Now add the parent's unique ID to each record, so that the parent relationship is now
    // independent of level.
    xNodes := JOIN(xNodes0, xNodes0, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId AND LEFT.level = RIGHT.level + 1
                                        AND LEFT.parentId = RIGHT.nodeId,
                                     TRANSFORM({TreeNodeDat, t_RecordID parentGuid},
                                                SELF.parentGuid := RIGHT.id;
                                                SELF := LEFT), LEFT OUTER, LOCAL);
    compressOneLevel(DATASET({xNodes}) cNodes, UNSIGNED tLevel) := FUNCTION
      // Find the nodes that need to be compressed out at this level
      compressNodes := SORT(cNodes(level = tLevel AND value = maxR8 AND number != 0 ), wi, treeId, id, LOCAL);
      // Find the children of the nodes to be compressed, and link them to the compressNode's parent
      // Note that there should only be one child for each compress node and it should be the left node,
      // but it needs to be also assigned the compressNode's isLeft.
      childNodes := SORT(cNodes(level = tLevel + 1), wi, treeId, parentGuid, LOCAL);
      replaceNodes := JOIN(compressNodes, childNodes, LEFT.wi = RIGHT.wi AND
                        LEFT.treeId = RIGHT.treeId AND LEFT.id = RIGHT.parentGuid,
                        TRANSFORM({compressNodes},
                                  SELF.parentGuid := LEFT.parentGuid,
                                  SELF.isLeft := LEFT.isLeft,
                                  SELF := RIGHT), LOCAL);
      // Eliminate the compressNodes and the child nodes and replace with the new child nodes
      outLevelNodes := cNodes(level = tLevel AND (value != maxR8 OR number = 0));
      outNextLevelNodes := JOIN(childNodes, replaceNodes, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId AND
                                                                    LEFT.id = RIGHT.id,
                                                          TRANSFORM(LEFT), LEFT ONLY, LOCAL);
      outNodes := outLevelNodes + replaceNodes + outNextLevelNodes + cNodes(level > tLevel + 1);
      RETURN outNodes;
    END; // CompressOneLevel
    newNodes := LOOP(xNodes, maxDepth, LEFT.level >= COUNTER,
                          compressOneLevel(ROWS(LEFT), COUNTER));
    // At this point, we have the correct set of nodes, but their level and local id's might be messed up.
    // They are linked by the global id, but we need to fix up all the levels and localid's by traversing
    // the tree from the top down using the global ids.
    fixupOneLevel(DATASET({newNodes}) fNodes, UNSIGNED tLevel) := FUNCTION
      // Start with the nodes for this level.  For the top-level (root), we use a different method
      // (i.e. no parent) in case the root was originally a compressed node and the current root therefore
      // has the level set wrong.
      fNodesS := SORT(fNodes, wi, treeId, parentGuid, -isLeft, LOCAL);
      levelNodes0 := IF(tLevel = 1, fNodesS(parentGuid = 0), fNodesS(level=tLevel));
      // Make sure that the level is set correctly for these items, and re-assign the local id (nodeId).
      levelNodes := PROJECT(levelNodes0, TRANSFORM({levelNodes0},
                        SELF.level := tLevel, SELF.nodeId := COUNTER, SELF := LEFT), LOCAL);
      // Now find all the children and set their level to tLevel + 1, and set the local parentId to the parent's nodeId
      fNodesS2:= SORT(fNodes, wi, treeId, parentGuid, LOCAL);
      nextLevelNodes := JOIN(levelNodes, fNodesS2, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId AND
                                LEFT.id = RIGHT.parentGuid,
                              TRANSFORM({levelNodes}, SELF.parentId := LEFT.nodeId, SELF.level := tLevel + 1,
                                SELF := RIGHT),
                              LOCAL);
      outNodes := levelNodes + nextLevelNodes + fNodes(level > tLevel + 1);
      RETURN outNodes;
    END; // fixupOneLevel
    outNodes := LOOP(newNodes, maxDepth, LEFT.level >= COUNTER,
                          fixupOneLevel(ROWS(LEFT), COUNTER));
    outNodesS := SORT(outNodes, wi, treeId, level, nodeId, LOCAL);
    RETURN PROJECT(outNodesS, TreeNodeDat);
  END; // CompressNodes
  /**
    * Extract the set of tree nodes from a model
    *
    */
  EXPORT DATASET(TreeNodeDat) Model2Nodes(DATASET(Layout_Model2) mod) := FUNCTION
    // Extract nodes from model as NumericField dataset
    nfNodes := ModelOps2.ToNumericField(mod, [FM1.nodes]);
    // Distribute by wi and id for distributed processing
    nfNodesD := DISTRIBUTE(nfNodes, HASH32(wi, id));
    nfNodesG := GROUP(nfNodesD, wi, id, LOCAL);
    nfNodesS := SORT(nfNodesG, wi, id, number);
    TreeNodeDat makeNodes(NumericField rec, DATASET(NumericField) recs) := TRANSFORM
      SELF.wi := rec.wi;
      SELF.treeId := recs[FMN3.treeId].value;
      SELF.level := recs[FMN3.level].value;
      SELF.nodeId := recs[FMN3.nodeId].value;
      SELF.parentId := recs[FMN3.parentId].value;
      SELF.isLeft := recs[FMN3.isLeft].value = 1;
      SELF.number := recs[FMN3.number].value;
      SELF.value := recs[FMN3.value].value;
      SELF.isOrdinal := recs[FMN3.isOrd].value = 1;
      SELF.depend := recs[FMN3.depend].value;
      SELF.support := recs[FMN3.support].value;
      SELF.ir := recs[FMN3.ir].value;
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
    nfSamples := ModelOps2.ToNumericField(mod, [FM1.samples]);
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
      indx1 := CHOOSE(c, FMN3.treeId, FMN3.level, FMN3.nodeId,
                         FMN3.parentId, FMN3.isLeft, FMN3.number,
                         FMN3.value, FMN3.isOrd,
                         FMN3.depend, FMN3.support, FMN3.ir);
      SELF.value := CHOOSE(c, d.treeId, d.level, d.nodeId, d.parentId,
                            (UNSIGNED)d.isLeft, d.number, d.value, (UNSIGNED)d.isOrdinal,
                            d.depend, d.support, d.ir);
      SELF.number := indx1;
      SELF.id := d.recordId;
    END;
    // Add a record id to nodes
    nodesExt := PROJECT(nodes, TRANSFORM({TreeNodeDat, UNSIGNED recordId}, SELF.recordId := COUNTER, SELF := LEFT));
    // Make into a NumericField dataset
    nfMod := NORMALIZE(nodesExt, 11, makeMod(LEFT, COUNTER));
    // Insert at position [modInd.nodes] in the ndArray
    mod := ModelOps2.FromNumericField(nfMod, [FM1.nodes]);
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
    indexes := ModelOps2.FromNumericField(nfIndexes, [FM1.samples]);
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

  /**
    * Compress and cleanup the model
    *
    * This function is provided to reduce the size of a model by compressing out
    * branches with only one child.  These branches are a result of the RF algorithm,
    * and do not affect the results of the model.
    * This is an expensive operation, which is why it is not done as a matter of
    * course.  It reduces the size of the model somewhat, and therefore slightly speeds
    * up any processing that uses the model, and reduces storage size.
    * You may want to compress the model if storage is at a premium, or if the model
    * is to be used many times (so that the slight performance gain is multiplied).
    * This also makes the model somewhat more readable, and could
    * be useful when analyzing the tree or converting it to another system
    * (e.g. LUCI) for processing.
    *
    */
  EXPORT DATASET(Layout_Model2) CompressModel(DATASET(Layout_Model2) mod) := FUNCTION
    nodes := Model2Nodes(mod);
    cNodes := CompressNodes(nodes);
    remainderMod := mod(indexes[1] != FM1.nodes);
    cMod := Nodes2Model(cNodes) + remainderMod;
    return cMod;
  END;

  // ModelStats
  EXPORT GetModelStats(DATASET(Layout_Model2) mod) := FUNCTION
    nodes := Model2Nodes(mod);
    treeStats := TABLE(nodes, {wi, treeId, nodeCount := COUNT(GROUP), depth := MAX(GROUP, level)},
                        wi, treeId);
    leafStats := TABLE(nodes(number=0), {wi, treeId, nodeCount := COUNT(GROUP), depth := AVE(GROUP, level),
                        totSupt := SUM(GROUP, support),
                        maxSupt := MAX(GROUP, support),
                        minDepth := MIN(GROUP, level)}, wi, treeId);
    treeSumm := TABLE(treeStats, {wi,
                        treeCount := COUNT(GROUP),
                        minTreeDepth := MIN(GROUP, depth),
                        maxTreeDepth := MAX(GROUP, depth),
                        avgTreeDepth := AVE(GROUP, depth),
                        minTreeNodes := MIN(GROUP, nodeCount),
                        maxTreeNodes := MAX(GROUP, nodeCount),
                        avgTreeNodes := AVE(GROUP, nodeCount),
                        totalNodes := SUM(GROUP, nodeCount)}, wi);
    leafSumm := TABLE(leafStats, {wi, treeCount := COUNT(GROUP),
                        avgLeafs := AVE(GROUP, nodeCount),
                        minSupport := MIN(GROUP, totSupt),
                        maxSupport := MAX(GROUP, totSupt),
                        avgSupport := AVE(GROUP, totSupt),
                        avgSupportPerLeaf := SUM(GROUP, totSupt) / SUM(GROUP, nodeCount),
                        maxSupportPerLeaf := MAX(GROUP, maxSupt),
                        avgLeafDepth := AVE(GROUP, depth),
                        minLeafDepth := MIN(GROUP, minDepth)}, wi);
    allStats := JOIN(treeSumm, leafSumm, LEFT.wi = RIGHT.wi, TRANSFORM(ModelStats,
                                                SELF := LEFT,
                                                SELF := RIGHT));
    RETURN allStats;
  END;
  /**
    * Feature Importance (intenal)
    *
    * Computes feature importance based on TreeNodeDat input
    *
    */
  EXPORT FeatureImportanceNodes(DATASET(TreeNodeDat) nodes) := FUNCTION
    treeCount := MAX(nodes, treeId);
    featureStats := TABLE(nodes(number > 0), {wi, number, importance := SUM(GROUP, ir * support)/treeCount,
                          uses := COUNT(GROUP)}, wi, number);
    fi := SORT(PROJECT(featureStats, TRANSFORM(FeatureImportanceRec, SELF := LEFT)), wi, -importance);
    RETURN fi;
  END;

  /**
    * Feature Importance
    *
    * Calculate feature importance using the Mean Decrease Impurity (MDI) method
    * from "Understanding Random Forests: by Gilles Loupe (https://arxiv.org/pdf/1407.7502.pdf)
    * and due to Breiman [2001, 2002]
    *
    * Each feature is ranked by:
    *   SUM for each branch node in which feature appears (across all trees):
    *     impurity_reduction * number of nodes split
    *
    */
  EXPORT FeatureImportance(DATASET(Layout_Model2) mod) := FUNCTION
    nodes := Model2Nodes(mod);
    fi := FeatureImportanceNodes(nodes);
    RETURN fi;
  END;
  // Extended tree node record
  SHARED xTreeNodeDat := RECORD(TreeNodeDat)
    SET OF UNSIGNED pathNodes;
  END;
  // Add to nodes a globally unique nodeId (i.e. 'id') as well as a set of ids for each
  // node representing the full path from the root, root and current inclusive.
  // Note that id only needs to be unique within a tree, so we use a LOCAL PROJECT.
  SHARED GetExtendedNodes(DATASET(TreeNodeDat) nodes) := FUNCTION
    nodesS := SORT(DISTRIBUTE(nodes, HASH32(wi, treeId)), wi, treeId, level, nodeId, LOCAL);
    // Calculate the ancestor id's one layer at a time from the root down.
    // Root just assigns its own id.  Others append their own id to the parent's id.
    xNodes0 := PROJECT(nodesS, TRANSFORM(xTreeNodeDat,
                                        SELF.id := COUNTER,
                                        SELF.pathNodes := [SELF.id],
                                        SELF := LEFT), LOCAL);
    loopBody(DATASET(xTreeNodeDat) n, UNSIGNED lev) := FUNCTION
      assignNodes := n(level = lev);
      parentNodes := n(level = lev - 1);
      newNodes := JOIN(assignNodes, parentNodes, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId AND
                                        LEFT.parentId = RIGHT.nodeId,
                                   TRANSFORM(xTreeNodeDat,
                                              SELF.pathNodes := RIGHT.pathNodes + LEFT.pathNodes,
                                              SELF := LEFT), LEFT OUTER, LOCAL);
      outNodes := n(level != lev) + newNodes;
      RETURN outNodes;
    END;
    maxLevel := MAX(nodesS, level);
    // Loop for each level
    xNodes := LOOP(xNodes0, maxLevel, loopBody(ROWS(LEFT), COUNTER));
    RETURN xNodes;
  END;
  SHARED dPath := RECORD
    t_Work_Item wi;
    t_RecordId id;
    t_TreeId treeId;
    SET OF UNSIGNED4 pathNodes;
  END;
  // Returns a record for each datapoint, for each tree, that includes the path for
  // the datapoint from the root to the leaf that it falls into.
  SHARED GetDecisionPaths(DATASET(TreeNodeDat) nodes, DATASET(GenField) X) := FUNCTION
    leafs := GetLeafsForData(nodes, X);
    xnodes := GetExtendedNodes(nodes);
    dps := JOIN(leafs, xnodes, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId AND LEFT.level = RIGHT.level AND
                                  LEFT.nodeId = RIGHT.nodeId,
                TRANSFORM(dPath,
                          SELF.wi := LEFT.wi,
                          SELF.id := LEFT.id,
                          SELF.treeId := LEFT.treeId,
                          SELF.pathNodes := RIGHT.pathNodes), LEFT OUTER);
    return dps;
  END;
  /**
    * Decision Distance Matrix
    *
    * Calculate a Decision Distance (DD) Matrix with one cell for each pair of points in [X1, X2].
    * If X2 is omitted, then the matrix will have one cell per pair of points in X1.
    * If X1 has N ids and X2 has M ids, then N x M records will be produced.
    * If only X1 is provided, then N x N records will be produced
    * This metric provides a number between zero and one, with zero indicating that the points
    * are very similar, and one representing maximal dissimilarity.
    * This is a distance measure within the decision space of the given random forest model.
    * 1 - DD conversely represents a similarity measure known as MeanSimilarityMeasure(MSM)
    *
    * DD(x1, x2) := 1 - MSM(x1, x2)
    * MSM(x1, x2) := MEAN for all trees (SM(tree, x1, x2)
    * SM(tree, x1, x2) := Maximum Level at which Path(tree, X1)
    *                     and Path(tree, X2) are equal / (|Path(tree, X1)| + |Path(tree, X2)| / 2)
    * Path(x) := The set of nodes from the root of the tree to the Leaf(x) inclusive.
    * |Path(x)| := The length of the set Path(x)
    *
    */
  EXPORT DecisionDistanceMatrix(DATASET(Layout_Model2) mod, DATASET(GenField) X1, DATASET(GenField) X2=empty_data) := FUNCTION
    nodes := Model2Nodes(mod);
    paths1 := SORT(GetDecisionPaths(nodes, X1), id);
    paths2 := IF(EXISTS(X2), SORT(GetDecisionPaths(nodes, X2), id), paths1);
    // Form an N x M upper triangular matrix for each tree, with values being the SM.
    sm := JOIN(paths1, paths2, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId,
                TRANSFORM({t_TreeId treeId, NumericField},
                          SELF.id := LEFT.id,
                          SELF.number := RIGHT.id,
                          SELF.value := int.CommonPrefixLen(LEFT.pathNodes, RIGHT.pathNodes) / ((COUNT(LEFT.pathNodes) + COUNT(RIGHT.pathNodes))/2),
                          SELF := LEFT));
    // Now average across trees
    // First redistribute by id, so that information for all trees for an id is on one node
    smD := DISTRIBUTE(sm, HASH32(wi, id));
    dd := TABLE(smD, {wi, id, number, REAL8 ddVal := 1 - AVE(GROUP, value)}, wi, id, number, LOCAL);
    // Return as NumericField array.
    ddm := PROJECT(dd, TRANSFORM(NumericField, SELF.value := LEFT.ddVal, SELF := LEFT), LOCAL);
    RETURN ddm;
  END; // DecisionDistanceMatrix
  /**
    * Uniqueness Factor
    *
    * Calculate how isolated each datapoint is in the decision space of the random forest
    * 0 < UF < 1, low values indicate that the datapoint is similar to other datapoints.
    * high values indicate uniqueness.
    *
    * Calculated as: SUM for all other points(Decision Distance) / (Number of Points - 1)
    *
    */
    EXPORT UniquenessFactor(DATASET(Layout_Model2) mod, DATASET(GenField) X1, DATASET(GenField) X2=empty_data) := FUNCTION
      // Get the Decision Distance Matrix
      ddm := DecisionDistanceMatrix(mod, X1, X2);
      ddmD := DISTRIBUTE(ddm, HASH32(wi, id));
      // Sum the distance for each point to every other point
      uf0 := TABLE(ddmD, {wi, id, totVal := AVE(GROUP, value)}, wi, id, LOCAL);
      uf := PROJECT(uf0, TRANSFORM(NumericField,
                                    SELF.number := 1,
                                    SELF.value := LEFT.totVal,
                                    SELF.wi := LEFT.wi,
                                    SELF.id := LEFT.id));
      RETURN uf;
    END; // Uniqueness Factor
END; // RF_Base
