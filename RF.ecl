IMPORT ML_Core as ML;
IMPORT ML.Types AS Types;
IMPORT std.system.Thorlib;
t_Work_Item := Types.t_Work_Item;
t_Count := Types.t_Count;
t_RecordId := Types.t_RecordID;
t_FieldNumber := Types.t_FieldNumber;
t_FieldReal := Types.t_FieldReal;
t_Discrete := Types.t_Discrete;
t_TreeId := t_FieldNumber;
t_NodeId := t_FieldNumber;
Layout_Model := Types.Layout_Model;
DiscreteField := Types.DiscreteField;
NumericField := Types.NumericField;

EXPORT RF := MODULE

  // New data types
  EXPORT GenField := RECORD
    t_Work_Item wi;
    t_RecordID id;
    t_FieldNumber number;
    t_FieldReal value;
    Boolean isOrdinal;
  END;
  SHARED treeVar := RECORD // Definition for each tree.
    t_TreeId treeId;
    t_FieldReal number; // X field number to associate with this tree
  END;
  // Tree node data
  // This is the major working structure for building the forest.
  // For efficiency and uniformity, this record structure serves several purposes
  // as the forest is built:
  // 1) It represents all of the X,Y data associated with each tree and node as the
  //   forest is being built.  This case is recognized by id > 0 (i.e. it is a data point)
  //   wi, treeId and NodeId represent the work-item and tree node with which the data is currently
  //         associated.
  //         All data in a trees sample is originally assigned to the root node (nodeId = 1)
  //         of its associated tree.
  //   id is the sample index in this trees data bootstrap sample
  //   origId is the sample index in the original X data.
  //   number is the field number from the X data
  //   isOrdinal indicates whether this data is Ordinal (true) or Nominal (false)
  //   value is the data value of this data point
  //   depend is the Y (dependent) value associated with this data point
  // 2) It represents the skeleton of the tree as the tree is built from the root down
  //   and the data points are subsumed (summarized) by the evolving tree structure.
  //   These cases can be identified by id = 0.
  //   2a) It represents branch (split) nodes:
  //       id = 0 -- All data was subsumed
  //       number > 0 -- The original field number of the X variable on which to split
  //       value -- the value on which to split
  //       parentId -- The nodeId of the branch at the previous level that leads to this
  //                   node.  Zero only for root.
  //       level -- The distance from the root (root = 1)
  //   2b) It represents leaf nodes:
  //       id = 0 -- All data was subsumed
  //       number = 0 -- This discriminates a leaf from a branch node
  //       depend has the Y value for that leaf
  //       parentId has the nodeId of the branch node at the previous level
  //       support has the count of samples that reached this leaf
  //       level -- The distance from the root (root = 1)
  // Each tree starts with all sampled data points assigned to the root node (i.e. level = 1, nodeId = 1)
  // As the trees grow, data points are assigned to deeper branches, and eventually to leaf nodes, where
  // they are ultimately subsumed (summarized) and removed from the dataset.
  // At the end of the forest growing process only the tree skeleton remains -- all the datapoints having
  // been summarized by the resulting branch and leaf nodes.
  SHARED TreeNodeDat := RECORD // Tree Node data.
  // Each record contains one independent variable sample for a given work-item, tree, and node
    t_TreeID treeId;
    t_NodeID nodeId;
    t_NodeID parentId;
    BOOLEAN  isLeft;             // If true, this is the paren'ts left split
    GenField;                    // Instance Independent Data - one attribute
    UNSIGNED2     level;         // Level of the node in tree.  Root is 1.
    t_Discrete    origId;        // The sample index (id) of the original X data that this sample came from
    t_FieldReal   depend;        // Instance Dependent value
    t_RecordId   support:=0;    // Number of data samples subsumed by this node
  END;

  // NodeSummary provides information to identify a given node
  SHARED NodeSummary := RECORD
    t_Work_Item wi;        // First five fields provide info about the node being split
    t_TreeID treeId;
    t_NodeID nodeId;
    t_NodeID parentId;     // Note that for any given (wi, treeId, nodeId, parentId and isLeft
                           //   will be constant, but we need to carry them through to maintain
                           //   the integrity of the nodes' relationships.
    BOOLEAN isLeft;
  END;
  // SplitDat is used to hold information about a potential split
  SHARED SplitDat := RECORD(NodeSummary)
    t_FieldNumber number;  // This is the field number that is being split
    t_FieldReal splitVal;  // This is the value at which to split <= splitval => LEFT >splitval
                           // => right
  END;

  // NodeImpurity carries identifying information for a node as well as its impurity level
  SHARED NodeImpurity := RECORD(NodeSummary)
    t_FieldReal impurity;  // The level of impurity of the given node.  Zero is most pure.
  END;

  // Information about each work item
  SHARED wiInfo := RECORD
    t_Work_Item   wi;
    t_RecordId    numSamples;       // Number of samples for this wi's data
    t_FieldNumber numFeatures;      // Number of features for this wi's data
    t_Count       featuresPerTree;  // Features per tree may be different for each work-item
                                    //   because it is base on numFeatures as well as the
                                    //   featuresPerTreeIn parameter to the module.
  END;
  // P log P calculation for entropy.  Note that shannon entropy uses log base 2 so the division by LOG(2) is
  // to convert the base from 10 to 2.
  SHARED P_Log_P(REAL P) := IF(P=1, 0, -P*LOG(P)/LOG(2));

  /**
    *
    */
  EXPORT RF_Any(DATASET(GenField) X_in,
                DATASET(GenField) Y_In,
                UNSIGNED numTrees=100,
                UNSIGNED featuresPerTreeIn=0,
                UNSIGNED maxDepth=255) := MODULE, VIRTUAL
    SHARED X := DISTRIBUTE(X_in, id);
    SHARED Y := DISTRIBUTE(Y_in, id);
    // Calculate work-item metadata
    Y_S := SORT(Y, wi);  // Sort Y by work-item
    // Each work-item needs its own metadata (i.e. numSamples, numFeatures, .  Construct that here.
    wiMeta0 := TABLE(X, {wi, numSamples := MAX(GROUP, id), numFeatures := MAX(GROUP, number), featuresPerTree := 0}, wi);
    wiInfo makeMeta(wiMeta0 lr) := TRANSFORM
      // If featuresPerTree was passed in as zero (default), use the square root of the number of features,
      // which is a good rule of thumb.  In general, with multiple work-items of different sizes, it is best
      // to default featuresPerTree, or set it to 1.
      fpt0 := IF(featuresPerTreeIn > 0, featuresPerTreeIn, TRUNCATE(SQRT(lr.numFeatures)));
      // In no case, let features per tree be greater than the number of features.
      SELF.featuresPerTree := MIN(fpt0, lr.numFeatures);
      SELF := lr;
    END;
    // Temp
    SHARED wiMeta := PROJECT(wiMeta0, makeMeta(LEFT));
    // Count of work-items
    SHARED wiCount := COUNT(wiMeta);
    // Number of features per tree
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
    SHARED maxSampleSize := MAX(wiMeta, numSamples);
    SHARED maxFeaturesPerTree := MAX(wiMeta, featuresPerTree);
    dummy := DATASET([{0,0,0}], sampleIndx);
    treeDummy := NORMALIZE(dummy, numTrees, TRANSFORM(sampleIndx, SELF.treeId := COUNTER, SELF := []));
    // Distribute by treeId to create the samples in parallel
    treeDummyD := DISTRIBUTE(treeDummy, treeId);
    // Now generate all samples in parallel
    SHARED treeSampleIndx :=NORMALIZE(treeDummyD, maxSampleSize, TRANSFORM(sampleIndx, SELF.origId := (RANDOM()%maxSampleSize) + 1, SELF.id := COUNTER, SELF := LEFT));

    SHARED DATASET(TreeNodeDat) SelectVarsForTrees(DATASET(TreeNodeDat) trees) := FUNCTION
      // At this point, trees should have one instance per tree per wi, distributed by (wi, treeId)
      // We are trying to choose featuresPerTree features out of the full set of features for each tree
      // Start by extending the tree data.  Add a random number field and create <features> records for each tree.
      xTreeNodeDat := RECORD(TreeNodeDat)
        UNSIGNED numFeatures;
        UNSIGNED featuresPerTree;
        UNSIGNED rnd;
      END;
      // Note that each work-item may have a different value for numFeatures and featuresPerTree
      xTreeNodeDat makeXTrees(treeNodeDat l, wiInfo r) := TRANSFORM
        SELF.numFeatures := r.numFeatures;
        SELF.featuresPerTree := r.featuresPerTree;
        SELF := l;
        SELF := [];
      END;
      xTrees := JOIN(trees, wiMeta, LEFT.wi = RIGHT.wi, makeXTrees(LEFT, RIGHT), LOOKUP, FEW);
      xTreeNodeDat getFeatures(xTreeNodeDat l, UNSIGNED c) := TRANSFORM
        // Choose twice as many as we need, so that when we remove duplicates, we will (almost always)
        // have at least the right number.  This is more efficient than enumerating all and picking <featuresPerTree>
        // from that set because numFeatures >> featuresPerTree.  We will occasionally get a tree that
        // has less than <featuresPerTree> variables, but that should only add to the diversity.
        nf := IF(c <= l.featuresPerTree*10, l.numFeatures, SKIP);
        SELF.number := (RANDOM()%nf) + 1;
        SELF.rnd := RANDOM();
        SELF := l;
      END;
      // Create twice as many features as we need, so that when we remove duplicates, we almost always
      // have at least as many as we need.
      varTrees0 := NORMALIZE(xTrees, maxFeaturesPerTree*10, getFeatures(LEFT, COUNTER));
      varTrees1 := SORT(varTrees0, wi, treeId, number, LOCAL);
      varTrees2 := DEDUP(varTrees1, wi, treeId, number, LOCAL);
      // Now we have up to <featuresPerTree> * 2 unique features per tree.  We need to whittle it down to
      // no more than <featuresPerTree>.
      varTrees3 := SORT(varTrees2, wi, treeId, rnd); // Mix up the features
      varTrees4 :=  GROUP(varTrees3, wi, treeId, LOCAL);
      // Filter out the excess vars and transform back to TreeNodeDat.  Set id (not yet used) just as an excuse
      // to check the count and skip if needed.
      varTrees := UNGROUP(PROJECT(varTrees4, TRANSFORM(TreeNodeDat, SELF.id := IF(COUNTER <= LEFT.featuresPerTree, 0, SKIP), SELF := LEFT)));
      // At this point, we have <featuresPerTree> records for almost every tree.  Occasionally one will have less
      // (but at least 1).
      RETURN ASSERT(varTrees, FALSE, 'varTrees: wi = ' + wi + ' treeId = ' + treeId + ' number = ' + number);
    END;

    SHARED DATASET(TreeNodeDat) SelectVarsForTreesw(DATASET(TreeNodeDat) trees) := FUNCTION
      varTrees := NORMALIZE(trees, featuresPerTreeIn, TRANSFORM(TreeNodeDat, SELF.number := COUNTER, SELF := LEFT));
      RETURN varTrees;
    END;

    // Sample with replacement <samples> items from X,Y for each tree and use those sample numbers
    // to choose values for each of the variables for each tree
    SHARED DATASET(TreeNodeDat) GetBootstrapForTree(DATASET(TreeNodeDat) treeVars) := FUNCTION
      // At this point, treeVars contains one record per tree per selected variable for each wi
      // Use the bootstrap (treeSampleIndxs) built at the module level

      // Note: At this point, treeVars and treeSampleIndx are both sorted and distributed by
      // treeId
      // We need to add the sample size from the wi to the dataset in order to filter appropriately
      xtv := RECORD(TreeNodeDat)
        t_RecordId numSamples;
      END;
      xTreeVars := JOIN(treeVars, wiMeta, LEFT.wi = RIGHT.wi, TRANSFORM(xtv, SELF.numSamples := RIGHT.numSamples,
                            SELF := LEFT), LOOKUP, FEW);
      // Expand the treeVars to include the sample index for each selected feature for each tree.
      // Size is now <numTrees> * <featuresPerTree> * <maxSamples> per wi
      // Note: this is a many to many join.
      treeVarDat0 := JOIN(xTreeVars, treeSampleIndx, LEFT.treeId = RIGHT.treeId,
                          TRANSFORM(xtv, SELF.origId := RIGHT.origId, SELF.id := RIGHT.id, SELF := LEFT),
                          MANY, LOOKUP);
      // Filter treeVarData0 to remove any samples with origId > numSamples for that wi.
      // The number of samples will not (in all cases) be = the desired sample size, but shouldn't create any bias.
      // This was our only need for numSamples, so we project back to TreeNodeDat format
      treeVarDat1 := PROJECT(treeVarDat0(origId <= numSamples), TreeNodeDat);
      // Now redistribute by <origId> to match the X,Y data
      treeVarDat1D := DISTRIBUTE(treeVarDat1, origId);
      // Use the indices to get the corresponding X value
      treeVarDat2 := JOIN(treeVarDat1D, X, LEFT.wi = RIGHT.wi AND LEFT.origId=RIGHT.id AND LEFT.number=RIGHT.number,
                          TRANSFORM(TreeNodeDat, SELF.value := RIGHT.value, SELF.isOrdinal := RIGHT.isOrdinal, SELF := LEFT),
                          LOCAL);
      // Now do the same for the corresponding Y (dependent) value
      // While we're at it, assign the data to the root (i.e. nodeId = 1, level = 1)
      treeVarDat := JOIN(treeVarDat2, Y, LEFT.wi = RIGHT.wi AND LEFT.origId=RIGHT.id,
                          TRANSFORM(TreeNodeDat, SELF.depend := RIGHT.value, SELF.nodeId := 1, SELF.level := 1, SELF := LEFT),
                          LOCAL);
      // At this point, we have one instance per tree per feature (selected for the tree) per sample, and each instance includes the corresponding X and Y values
      //  (i.e. value and depend), for each work unit.  treeVarDat is distributed by sample id.
      RETURN treeVarDat;
    END;

    // Create the set of tree definitions -- One single node per tree (the root), with all tree samples associated with that root.
    SHARED DATASET(TreeNodeDat) InitTrees := FUNCTION
      // Create an empty tree data instance per work-item
      dummyTrees := PROJECT(wiMeta, TRANSFORM(TreeNodeDat, SELF.wi := LEFT.wi, SELF := []));
      // Use that to create "numTrees" dummy trees -- a dummy (empty) forest per wi
      trees := NORMALIZE(dummyTrees, numTrees, TRANSFORM(TreeNodeDat, SELF.treeId:=COUNTER, SELF.wi := LEFT.wi, SELF:=[]));
      // Distribute by wi and treeId
      treesD := DISTRIBUTE(trees, HASH32(wi, treeId));
      // Now, choose a random subset of features for each tree
      featureTrees := SelectVarsForTrees(treesD);
      // Now, choose bootstrap sample of X,Y for each tree
      roots := GetBootstrapForTree(featureTrees);
      // At this point, each tree is fully populated with a single root node(i.e. 1).  All the data is associated with the root node.
      // Roots has each tree's bootstrap sample of each dependent variable (selected for the tree) as well as the corresponding dependent variable value.
      // Roots is distributed by origId (original sample index)
      RETURN roots;
    END;

    SHARED VIRTUAL DATASET(NodeImpurity) CalcNodeImpurity(DATASET(TreeNodeDat) NodeDat) := FUNCTION
      return DATASET([], NodeImpurity);
    END;

    SHARED VIRTUAL DATASET(NodeImpurity) IsPureEnough(DATASET(NodeImpurity) nodeImp) := FUNCTION
      return DATASET([], NodeImpurity);
    END;

    SHARED VIRTUAL DATASET(TreeNodeDat) GrowForestLevel(DATASET(TreeNodeDat) nodes, t_Count treeLevel) := FUNCTION
      return DATASET([], TreeNodeDat);
    END;
    // Grow one layer of the forest

    // Grow a Classification Forest from a set of roots containing all the data points (X and Y) for each tree.
    SHARED DATASET(TreeNodeDat) GrowForestC(DATASET(TreeNodeDat) roots) := FUNCTION
      // Localize all the data by tree and node
      rootsD := DISTRIBUTE(roots, HASH32(treeId, NodeId));
      // Grow the forest one level at a time.
      treeNodes  := LOOP(rootsD, LEFT.id > 0, (COUNTER <= maxDepth) AND EXISTS(ROWS(LEFT)) , GrowForestLevel(ROWS(LEFT), COUNTER));
      return treeNodes;
    END;

    // Generate all tree nodes for classification
    SHARED DATASET(TreeNodeDat) GetNodesC := FUNCTION
      // First create a diverse set of tree roots
      // with a random selection of features, and a unique bootstrap sample out of X and Y
      roots := InitTrees;
      // We now have a root node for each tree (leve = 1, nodeId = 1).  All of the data is
      // associated with the root for each tree.
      // Now we want to grow each tree by adding nodes, and moving the data
      // points to lower and lower nodes at each split.
      // When we are done, all of the data will be gone and all that will remain
      // is the skeleton of the decision tree with splits and leaf nodes.
      forestNodes := GrowForestC(roots);
      // We now just have the structure of the decision tree remaining.  All data
      // is now summarized by the tree's structure into leaf nodes.
      RETURN forestNodes;
    END;

    // Generate all tree nodes for regression
    SHARED DATASET(TreeNodeDat) GetNodesR := FUNCTION
      // Temp
      RETURN GetNodesC;
    END;

    // Convert a model to a set of tree nodes
    SHARED DATASET(TreeNodeDat) Model2Nodes(DATASET(Layout_Model) mod) := FUNCTION
      // Temp
      RETURN GetNodesC;
    END;

    SHARED DATASET(Layout_Model) Nodes2Model(DATASET(TreeNodeDat) nodes) := FUNCTION
      // Temp
      RETURN DATASET([{0,0,0,0}], Layout_Model);
    END;

    // Produce a class for each X sample given an expanded forest model (set of tree nodes)
    // TEMP EXPORT
    EXPORT DATASET(DiscreteField) ForestClassify(DATASET(TreeNodeDat) tNodes, DATASET(GenField) X) := FUNCTION
      // Replicate each X sample to all nodes
      tNodesD := DISTRIBUTE(tNodes, HASH32(wi, treeid));
      nodeCount := Thorlib.nodes();
      xD0 := DISTRIBUTE(X, HASH32(wi, id));
      xD1 := NORMALIZE(xD0, nodeCount, TRANSFORM({GenField, UNSIGNED thornode}, SELF.thornode := COUNTER, SELF := LEFT));
      xD2 := DISTRIBUTE(xD1, thornode);
      xD := PROJECT(xD2, GenField);
      // Extend each root for each ID in X
      roots := tNodesD(level = 1);
      rootsExt := JOIN(roots, xD(number=1), LEFT.wi = RIGHT.wi, TRANSFORM(TreeNodeDat, SELF.id := RIGHT.id, SELF := LEFT),
                        LEFT OUTER, LOCAL);
      rootBranches := rootsExt(number != 0); // Roots are almost always branch (split) nodes.
      rootLeafs := rootsExt(number = 0); // Unusual but not impossible
      loopBody(DATASET(TreeNodeDat) levelBranches, UNSIGNED tLevel) := FUNCTION
        // At this point, we have one record per node, per id.
        // We extend each id down the tree one level at a time, picking the correct next nodes
        // for that id at each branch.
        // Next nodes are returned -- both leafs and branches.  The leafs are filtered out by the LOOP,
        // while the branches are send on to the next round.
        // Ultimately, a leaf is returned for each id, which defines our final result.
        // Select the next nodes by combining the selected data field with each node
        // Note that:  1) we retain the id from the previous round, but the field number(number) is derived from the branch
        //             2) 'value' in the node is the value to split upon, while value in the data (X) is the value of that datapoint
        //             3) NodeIds at level n + 1 are deterministic.  The child nodes at the next level's nodeId is 2 * nodeId -1 for the
        //                left node, and 2 * nodeId for the right node.
        branchVals := JOIN(levelBranches, xD, LEFT.wi = RIGHT.wi AND LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number,
                            TRANSFORM({TreeNodeDat, BOOLEAN branchLeft},
                                        SELF.branchLeft :=  ((RIGHT.isOrdinal AND RIGHT.value <= LEFT.value) OR
                                                            ((NOT RIGHT.isOrdinal) AND RIGHT.value = LEFT.value)),
                                        SELF.parentId := LEFT.nodeId, SELF := LEFT),
                            LOCAL);
        // Now, nextNode indicates the selected (left or right) nodeId at the next level for each branch
        // Now we use nextNode to select the node for the next round, for each instance
        nextLevelNodes := tNodesD(level = tLevel + 1);
        nextLevelSelNodes := JOIN(branchVals, nextLevelNodes, LEFT.wi = RIGHT.wi AND
                                LEFT.treeId = RIGHT.treeId AND LEFT.nodeId = RIGHT.parentId AND
                                LEFT.branchLeft = RIGHT.isLeft,
                                TRANSFORM(TreeNodeDat, SELF.id := LEFT.id, SELF := RIGHT), LOCAL);
        // Return the selected nodes at the next level.  These nodes may be leafs or branches.
        // Any leafs will be filtered out by the loop.  Any branches will go on to the next round.
        // When there are no more branches to process, we are done.  The selected leafs for each datapoint
        // is returned.
        RETURN nextLevelSelNodes;
      END;
      // The loop will return the deepest leaf node associated with each sample.
      selectedLeafs0 := LOOP(rootBranches, LEFT.number>0, EXISTS(ROWS(LEFT)),
                            loopBody(ROWS(LEFT), COUNTER));
      selectedLeafs := selectedLeafs0 + rootLeafs;
      // At this point, we have one leaf node per tree per datapoint (X)
      // The leaf nodes contain the final class in their 'depend' field.
      // Now we need to vote among the trees, with the mode (most common value (i.e. mode) winning)
      // First redistribute by wi and ID
      selectedLeafsD := DISTRIBUTE(selectedLeafs, HASH32(wi, id));
      selectedLeafsS := SORT(selectedLeafsD, wi, id, depend, LOCAL);
      selectedLeafCounts := TABLE(selectedLeafS, {wi, id, depend, cnt := COUNT(GROUP)}, wi, id, depend, LOCAL);
      // Now one record per datapoint per value of depend (Y) with the count of 'votes' for each depend value
      // Reduce to one record per datapoint, and convert to DiscreteField format.
      selectedLeafCountsD := DISTRIBUTE(selectedLeafCounts, HASH32(wi, id));
      selectedLeafCountsS := SORT(selectedLeafCountsD, wi, id, -cnt, LOCAL); // LOCAL!!!
      // Keep the first leaf value for each wi and id.  That is the one with the highest count (vote)
      selectedClasses := DEDUP(selectedLeafCountsS, wi, id, LOCAL);
      results := PROJECT(selectedClasses, TRANSFORM(DiscreteField, SELF.number := 1, SELF.value := LEFT.depend, SELF := LEFT));
      RETURN results;
    END;

    // Predict a Y value for each X sample, given an expanded forest model
    SHARED DATASET(NumericField) ForestPredict(DATASET(TreeNodeDat) tNodes, DATASET(GenField) X) := FUNCTION
      RETURN DATASET([{0,0,0,0}], NumericField);
    END;

    // Get classification model
    EXPORT GetModelC := FUNCTION
      nodes := GetNodesC;
      mod := Nodes2Model(nodes);
      RETURN nodes; // Temp
    END;

    // Get regression model
    EXPORT GetModelR := FUNCTION
      nodes := GetNodesR;
      mod := Nodes2Model(nodes);
      RETURN mod;
    END;

    SHARED empty_model := DATASET([], Layout_Model);
    // Create a classification forest from the X, Y data, or use model passed in.
    // Use that forest model to predict the Y for a set of X values.
    EXPORT Classify(DATASET(GenField) X, DATASET(Layout_Model) mod=empty_model) := FUNCTION
      tNodes := IF(EXISTS(mod), Model2Nodes(mod), getNodesC);
      classes := ForestClassify(tNodes, X);
      RETURN classes;
    END;

    EXPORT Predict(DATASET(GenField) X, DATASET(Layout_Model) mod=empty_model) := FUNCTION
      tNodes := IF(EXISTS(mod), Model2Nodes(mod), getNodesR);
      classes := ForestPredict(tNodes, X);
      RETURN classes;
    END;
  END;
  // Module for Classification Forest
  EXPORT RF_Classification(DATASET(GenField) X_in,
                            DATASET(GenField) Y_In,
                            UNSIGNED numTrees=100,
                            UNSIGNED featuresPerTree=0,
                            UNSIGNED maxDepth=255) := MODULE(RF_Any(X_in, Y_in, numTrees, featuresPerTree, maxDepth))
    // Find the best split for a given set of nodes.  In this case, it is the one with the highest information
    // gain.  Every possible split point is considered for each independent variable in the tree.
    // For nominal variables, the split is an equality split on one of the possible values for that variable
    // (i.e. split into = s and != s).  For ordinal variables, the split is an inequality (i.e. split into <= s and > s)
    // For each node, the split with the highest Information Gain Ratio (IGR) is returned.
    SHARED DATASET(SplitDat) findBestSplit(DATASET(TreeNodeDat) nodeDat, DATASET(NodeImpurity) parentEntropy) := FUNCTION
      // Calculate the Information Gain Ratio (IGR) for each split.
      // IGR := Information Gain (IG) / Intrinsic Value (IV)  // Note: Intrinsic Value is also know as:
      //         Split Info -- the info content of the X variable
      // IG := Entropy(H) of Parent - Entropy(H) of the proposed split := H-parent - SUM(prob(child) * H-child) for each child group of the split
      // IV := -SUM(Prob(x) * Log2(Prob(x)) for all values of X independent variable
      // H := -SUM(Prob(y) * Log2(Prob(y)) for all values of Y dependent variable

      // Calculate the Intrinsic Value (IV) for each feature that we can split on.
      featureVals := TABLE(nodeDat, {wi, treeId, nodeId, number, value, isOrd := MAX(GROUP, isOrdinal), cnt := COUNT(GROUP)},
                            wi, treeId, nodeId, number, value, LOCAL);
      features := TABLE(featureVals, {wi, treeId, nodeId, number, tot := SUM(GROUP, cnt), gmax := MAX(GROUP, value)},
                            wi, treeId, nodeId, number, LOCAL);
      featureVals2 := JOIN(featureVals, features, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId
                              AND LEFT.nodeId = RIGHT.nodeId AND LEFT.number = RIGHT.number,
                            TRANSFORM({featureVals, REAL prop, REAL plogp, t_FieldReal gmax, UNSIGNED tot},
                                        SELF.prop := LEFT.cnt / RIGHT.tot, SELF.plogp := P_Log_P(SELF.prop),
                                        SELF.gmax := RIGHT.gmax, SELF.tot := RIGHT.tot, SELF := LEFT), LOCAL);
      featureIVs := TABLE(featureVals2, {wi, treeId, nodeId, number, iv := SUM(GROUP, plogp)},
                            wi, treeId, nodeId, number, LOCAL);
      // Replicate each datapoint for the node to every possible split for that node
      // Mark each datapoint as being left or right of the split.  Handle both Ordinal and Nominal cases.
      // First, filter the feature values so that we don't replicate data for the last data-point, except
      // for nominal features with more than two values.  This is strictly an optimization, since for
      // binary nominals, splitting on one value is the same as splitting on the other, and for ordinals,
      // if the last value is used, it will not result in any information gain since all data will be
      // to the left of the split.
      featureVals3 := featureVals2((NOT isOrd AND tot > 2) OR value != gmax);
      allSplitDat := JOIN(nodeDat, featureVals3, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId
                          AND RIGHT.nodeId = LEFT.nodeId AND LEFT.number = RIGHT.number,
                        TRANSFORM({TreeNodeDat, t_FieldReal splitVal}, SELF.splitVal := RIGHT.value,
                                  SELF.isLEFT := IF((LEFT.isOrdinal AND LEFT.value <= SELF.splitVal)
                                                    OR (NOT LEFT.isOrdinal AND LEFT.value = SELF.splitVal),TRUE, FALSE),
                                  SELF := LEFT), FULL OUTER, LOCAL);
      // Calculate the entropy of the left and right groups of each split
      // Group by value of Y (depend) for left and right splits
      dependGroups := TABLE(allSplitDat, {wi, treeId, nodeId, number, splitVal, isLeft, depend,
                                UNSIGNED cnt := COUNT(GROUP)},
                              wi, treeId, nodeId, number, splitVal, isLeft, depend, LOCAL);
      // Sum up the number of data points for left and right splits
      dependSummary := TABLE(dependGroups, {wi, treeId, nodeId, number, splitVal, isLeft,
                              UNSIGNED tot := SUM(GROUP, cnt)},
                              wi, treeId, nodeId, number, splitVal, isLeft, LOCAL);
      // Calculate p_log_p for each Y value for left and right splits
      dependRatios := JOIN(dependGroups, dependSummary,
                         LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId AND LEFT.nodeId = RIGHT.nodeId AND
                            LEFT.number = RIGHT.number AND LEFT.splitVal = RIGHT.splitVal
                            AND LEFT.isLeft = RIGHT.isLeft,
                         TRANSFORM({dependGroups, REAL prop, REAL plogp},
                            SELF.prop := LEFT.cnt / RIGHT.tot, SELF.plogp := P_Log_P(SELF.prop),
                            SELF := LEFT),
                            LOCAL);
      // Sum the p_log_p's for each Y value to get the entropy of the left and right splits.
      entropies := TABLE(dependRatios, {wi, treeId, nodeId, number, splitVal, isLeft, tot := SUM(GROUP, cnt),
                              entropy := SUM(GROUP, plogp)},
                            wi, treeId, nodeId, number, splitVal, isLeft, LOCAL);
      // Now sum the weighted entropies of the two groups (weighted by number of datapoints in each)
      // Note that 'tot' is number of datapoints for each side of the split.
      totalSplitEntropy := TABLE(entropies, {wi, treeId, nodeId, number, splitVal,
                                  REAL totEntropy := SUM(GROUP, entropy*tot) / SUM(GROUP,tot)},
                                wi, treeId, nodeId, number, splitVal, LOCAL);
      // Now calculate Information Gain
      ig := JOIN(totalSplitEntropy, parentEntropy, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId AND
                    LEFT.nodeId = RIGHT.nodeId,
                  TRANSFORM({totalSplitEntropy, t_NodeID parentId, BOOLEAN isLeft, REAL ig},
                            SELF.ig := RIGHT.impurity - LEFT.totEntropy,
                            SELF.parentId := RIGHT.parentId, SELF.isLeft := RIGHT.isLeft, SELF := LEFT), LOCAL);
      // Now calculate the Information Gain Ratio
      // In order to stop the tree-building process when there is no split that gives information-gain
      // we set number to zero to indicate that there is no best split when we hit that case.
      // That happens when the data is not fully separable by the independent variables.
      igr := JOIN(ig, featureIVs, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId AND LEFT.nodeId = RIGHT.nodeId
                    AND LEFT.number = RIGHT.number,
                TRANSFORM({ig, REAL igr}, SELF.igr := LEFT.ig / RIGHT.iv,
                          SELF.number := IF(SELF.igr > 0, LEFT.number, 0), SELF := LEFT), LOCAL);
      // Use IGR to find the best split for each node (i.e. the one with the highest information gain ratio)
      igrS := SORT(igr, wi, treeId, nodeId, -igr, LOCAL);
      bestSplits := DEDUP(igrS, wi, treeId, nodeId, LOCAL);
      RETURN PROJECT(bestSplits, SplitDat);
    END;

    // Grow one layer of the forest
    SHARED DATASET(TreeNodeDat) GrowForestLevel(DATASET(TreeNodeDat) nodes, t_Count treeLevel) := FUNCTION
      // Calculate the Impurity for each node.  Since we have Y associated with each feature,
      // we need to dedup the features to get a clean Y for each tree.  Is there a better way to handle this?
      nodesS := SORT(nodes, wi, treeId, nodeId, id, number, value, LOCAL);
      // NodeY has one record per Y value per node (i.e. duplicate Y's (for each field number) have been removed)
      nodeY := DEDUP(nodes, wi, treeId, nodeId, id, LOCAL);
      // NodeValCounts has one record per node, per value of the depenent variable (Y)
      nodeValCounts := TABLE(nodeY, {wi, treeId, nodeId, depend, parentId, isLeft, cnt:= COUNT(GROUP)},
                              wi, treeId, nodeId, depend, parentId, isLeft, LOCAL);
      // NodeCounts is the count of data items for the node
      nodeCounts := TABLE(nodeValCounts, {wi, treeId, nodeId, tot:= SUM(GROUP, cnt)},
                              wi, treeId, nodeId, LOCAL);
      // Now we can calculate the information entropy for each node
      // Entropy is defined as SUM(plogp(proportion of each Y value)) for each Y value
      nodeEntInfo := JOIN(nodeValCounts, nodeCounts, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId AND
                        LEFT.nodeId = RIGHT.nodeId,
                      TRANSFORM({nodeValCounts, REAL4 prop, REAL4 plogp}, SELF.prop:= LEFT.cnt/RIGHT.tot, SELF.plogp:= P_LOG_P(LEFT.cnt/RIGHT.tot),
                                , SELF:=LEFT), LOCAL);
      // Note that for any (wi, treeId, nodeId), parentId and isLeft will be constant, but we need to carry
      //   them forward.
      nodeEnt0 := TABLE(nodeEntInfo, {wi, treeId, nodeId, parentId, isLeft,
                                         entropy := SUM(GROUP, plogp)}, wi, treeId, nodeId, parentId, isLeft, LOCAL);
      nodeImp := PROJECT(nodeEnt0, TRANSFORM(NodeImpurity, SELF.impurity := LEFT.entropy, SELF := LEFT));

      // Filtering pure and non-pure nodes. We translate any pure nodes and their associated data into a leaf node.
      // Impure nodes need further splitting, so they are passed into the next phase.
      pureEnough := nodeImp(impurity = 0);  // Zero entropy indicates complete purity

      // Eliminate any data associated with the leafNodes from the original node data.  What's left
      // is the data for the impure nodes that still need to be split
      toSplit := JOIN(nodesS, pureEnough, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId AND LEFT.nodeId = RIGHT.nodeId,
                      TRANSFORM(LEFT),
                      LEFT ONLY, LOCAL);
      // Find the best split points (field number, split value) for each node.
      bestSplits0 := findBestSplit(toSplit, nodeImp);
      // Handle special case when we hit maxDepth for the tree.  Mark every split as bad so that
      // all the reamaining data will be treated as mixedLeafs.
      bestSplits := IF(treeLevel < maxDepth, bestSplits0,
                      PROJECT(bestSplits0, TRANSFORM(SplitDat, SELF.number := 0, SELF := LEFT)));
      // Reasonable splits were found
      goodSplits := bestSplits(number != 0);
      // No split made any progress, or we are at maxDepth for the tree
      badSplits := bestSplits(number = 0);
      // Take any bad split data out of toSplit, and call it toSplit2
      toSplit2 := JOIN(toSplit, goodSplits, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId AND
                        LEFT.nodeId = RIGHT.nodeId, TRANSFORM(LEFT), LOCAL);
      // Create a split node and two child nodes for each split. Move the data to the child nodes.
      // First move the data to new child nodes.
      // Start by finding the data samples that fit into the left and the right
      leftIds := JOIN(toSplit2, goodSplits, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId AND
                          LEFT.nodeId = RIGHT.nodeId AND LEFT.number = RIGHT.number AND
                          ((LEFT.isOrdinal AND LEFT.value <= RIGHT.splitVal) OR
                            (NOT LEFT.isOrdinal AND LEFT.value = RIGHT.splitVal)),
                        TRANSFORM({t_Work_Item wi, t_TreeId treeId, t_NodeId nodeId, t_RecordId id},
                          SELF.treeId := LEFT.treeId, SELF.nodeId := LEFT.nodeId, SELF.id := LEFT.id, SELF := LEFT),
                        LOCAL);
      // LeftData contains all of the node data for the left split (i.e. for Ordinal data: where val <= splitVal,
      //  for Nominal data: where val = splitVal)
      // RightData contains all the node data for the right split(i.e. for Ordinal data: where val > splitVal,
      //  for Nominal data: where val <> splitVal)
      // Move the left and right data to new left and right nodes at the next level
      // Get ids for left nodes of the split.  Note that nodeIds only need to be unique within a level.
      // Left ids are assigned every other value (1, 3, 5, ...) to leave room for the rights,
      // which will be left plus 1 for a given parent node.  This provides an inexpensive way to assign
      // ids at the next level (though it opens the door for overflow of nodeId).  We handle that
      // case later.
      leftData := JOIN(toSplit2, leftIds, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId AND
                        LEFT.nodeId = RIGHT.nodeId AND LEFT.id = RIGHT.id,
                        TRANSFORM(TreeNodeDat, SELF.level := treeLevel + 1, SELF.nodeId := LEFT.nodeId * 2 - 1,
                                  SELF.parentId := LEFT.nodeId, self.isLeft := TRUE,
                                  SELF := LEFT), LOCAL);
      rightData := JOIN(toSplit2, leftIds, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId AND
                        LEFT.nodeId = RIGHT.nodeId AND LEFT.id = RIGHT.id,
                         TRANSFORM(TreeNodeDat, SELF.level := treeLevel + 1, SELF.nodeId := LEFT.nodeId * 2,
                                  SELF.parentId := LEFT.nodeId, self.isLeft := FALSE,
                                  SELF := LEFT), LEFT ONLY, LOCAL);
      nextLevelDat0 := SORT(leftData + rightData, wi, treeId, nodeId, LOCAL);
      // Occasionally, recalculate the nodeIds to make them contiguous to avoid an overflow
      // error when the trees get very deep.  Note that nodeId only needs to be unique within
      // a level.  It is not required that they be a function of the parent's id since parentId will
      // anchor the child to its parent.
      nextLevelIds := TABLE(nextLevelDat0, {wi, treeId, nodeId, t_NodeID newId := 0}, wi, treeId, nodeId, LOCAL);
      nextLevelIdsG := GROUP(nextLevelIds, wi, treeId, LOCAL);
      newIdsG := PROJECT(nextLevelIdsG, TRANSFORM({nextLevelIds}, SELF.newId := COUNTER, SELF := LEFT));
      newIds := UNGROUP(newIdsG);
      fixupIds := SORT(JOIN(nextLevelDat0, newIds, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId AND
                            LEFT.nodeId = RIGHT.nodeId,
                        TRANSFORM(TreeNodeDat, SELF.nodeId := RIGHT.newId, SELF := LEFT), LOCAL), wi, treeId, nodeId, LOCAL);
      nextLevelDat := IF(treeLevel % 32 = 0, fixupIds, nextLevelDat0); // Recalculate every 32 levels to avoid overflow
      // Now reduce each splitNode to a single skeleton node with no data.
      // For a split node (i.e. branch), we only use treeId, nodeId, number (the field number to split on), value (the value to split on), and parent-id
      splitNodes := PROJECT(goodSplits, TRANSFORM(TreeNodeDat, SELF.level := treeLevel, SELF.wi := LEFT.wi,
                            SELF.treeId := LEFT.treeId,
                            SELF.nodeId := LEFT.nodeId, self.number := LEFT.number, self.value := LEFT.splitVal,
                            SELF.parentId := LEFT.parentId,
                            SELF.isLeft := LEFT.isLeft,
                            SELF := []));
      // Now handle the leaf nodes, which are the pure-enough nodes, plus the bad splits (i.e. no good
      // split left).
      // Handle the badSplit case: there's no feature that will further split the data = mixed leaf node.
      // Classify the point according to the most frequent class, and create a leaf node to summarize it.
      mixedLeafs0 := JOIN(nodeValCounts, badSplits, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId AND
                          LEFT.nodeId = RIGHT.nodeId,
                        TRANSFORM(TreeNodeDat, SELF.wi := LEFT.wi, SELF.level := treeLevel,
                                SELF.treeId := LEFT.treeId, SELF.nodeId := LEFT.nodeId,
                                SELF.parentId := LEFT.parentId, SELF.isLeft := LEFT.isLeft, SELF.id := 0, SELF.number := 0,
                                SELF.depend := LEFT.depend, SELF.support := LEFT.cnt, SELF := []), LOCAL);
      mixedLeafs1 := SORT(mixedLeafs0, wi, treeId, nodeId, -support, LOCAL);
      mixedLeafs := DEDUP(mixedLeafs1, wi, treeId, nodeId, LOCAL); // Finds the most common value
      // Create a single leaf node instance to summarize each pure node's data
      // The leaf node instance only has a few significant attributes:  The tree and node id,
      // the dependent value, and the level, as well
      // as the support (i.e. the number of datapoints that fell into that leaf).
      pureNodes := JOIN(nodeValCounts, pureEnough, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId AND
                          LEFT.nodeId = RIGHT.nodeId,
                        TRANSFORM(TreeNodeDat, SELF.wi := LEFT.wi, SELF.level := treeLevel,
                                SELF.treeId := LEFT.treeId, SELF.nodeId := LEFT.nodeId,
                                SELF.parentId := LEFT.parentId,
                                SELF.isLeft := LEFT.isLeft, SELF.id := 0, SELF.number := 0,
                                SELF.depend := LEFT.depend, SELF.support := LEFT.cnt, SELF := []), LOCAL);
      leafNodes := pureNodes + mixedLeafs;
      // Return the three types of nodes: leafs at this level, splits (branches) at this level, and nodes at the next level (children of the branches).
      RETURN leafNodes + splitNodes + nextLevelDat;
    END;
  END;
END;