/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2017 HPCC Systems®.  All rights reserved.
############################################################################## */
IMPORT $.^ AS LT;
IMPORT LT.internal AS int;
IMPORT LT.LT_Types AS Types;
IMPORT ML_Core as ML;
IMPORT ML.Types AS CTypes;
IMPORT std.system.Thorlib;

GenField := Types.GenField;
TreeNodeDat := Types.TreeNodeDat;
SplitDat := Types.SplitDat;
NodeImpurity := Types.NodeImpurity;
wiInfo := Types.wiInfo;
t_Work_Item := CTypes.t_Work_Item;
t_Count := CTypes.t_Count;
t_RecordId := CTypes.t_RecordID;
t_FieldNumber := CTypes.t_FieldNumber;
t_TreeId := t_FieldNumber;
Layout_Model2 := CTypes.Layout_Model2;
t_FieldReal := CTypes.t_FieldReal;
t_NodeId := Types.t_NodeId;
NumericField := CTypes.NumericField;

/**
  * Module for Regression Forest
  *
  * This module provides a Random Forest Regressor based on Breiman, 2001
  * with extensions.
  *
  * See RF_Base for a description of the Theory of Operation of this module.
  */
EXPORT RF_Regression(DATASET(GenField) X_in=DATASET([], GenField),
                          DATASET(GenField) Y_In=DATASET([], GenField),
                          UNSIGNED numTrees=100,
                          UNSIGNED featuresPerNode=0,
                          UNSIGNED maxDepth=255) := MODULE(int.RF_Base(X_in, Y_in, numTrees, featuresPerNode, maxDepth))
  SHARED MinVarRed := .000001;  // Minimum variance reduction to consider a split useful
  SHARED PureNodeThreshold := .000001; // Impurity below this level is considered pure.
  // Find the best split for a given set of nodes.  In this case, it is the one with the greatest reduction in variance.
  // Every possible split point is considered for each independent variable in the tree.
  // For nominal variables, the split is an equality split on one of the possible values for that variable
  // (i.e. split into = s and != s).  For ordinal variables, the split is an inequality (i.e. split into <= s and > s)
  // For each node, the split with the highest reduction in variance is returned.
  SHARED DATASET(SplitDat) findBestSplit(DATASET(TreeNodeDat) nodeDat, DATASET(NodeImpurity) parentSummary) := FUNCTION
    // Calculate the Variance Reduction (VR) for each split.
    // VR := ParentVariance(PV) - SplitVariance(SV)
    // SV := (Variance(LeftData) * COUNT(LeftData) + Variance(RightData) * COUNT(RightData)) / COUNT(AllData)

    // Start with all values of each feature for each node.  altVal will be used later.
    featureVals := TABLE(nodeDat, {wi, treeId, nodeId, number, value, isOrdinal, cnt := COUNT(GROUP), REAL8 altVal := 0},
                          wi, treeId, nodeId, number, value, isOrdinal, LOCAL);
    // Set altVal to the midpoint between feature values (for ordered values) or 'value' otherwise.
    // Note that the first value for each ordered feature will end up with altVal = value, since there
    // is no LEFT record.  These will be eliminated later as they are not a valid split point.
    featureValsS := SORT(featureVals, wi, treeId, nodeId, number, value, LOCAL);
    featureValsM := ITERATE(featureValsS, TRANSFORM({featureVals},
                              SELF.altVal := IF(RIGHT.wi = LEFT.wi AND LEFT.treeId = RIGHT.treeId AND
                                              LEFT.nodeId = RIGHT.nodeId AND LEFT.number = RIGHT.number AND RIGHT.isOrdinal,
                                             // LEFT.value + .00001, RIGHT.value),
                                              (LEFT.value + RIGHT.value)/2, RIGHT.value),
                              SELF := RIGHT), LOCAL);
    features := TABLE(featureVals, {wi, treeId, nodeId, number, isOrdinal, vals := COUNT(GROUP)},
                          wi, treeId, nodeId, number, isOrdinal, LOCAL);
    // We want to eliminate constant features (i.e. features with only one value for a node) from
    // consideration.  But we have to guard against the case where all of the selected features are
    // constant.  To do that, we save one constant feature per node in case we have to resort to
    // using it later.
    constantFeatures := features(vals = 1);
    dummySplits := DEDUP(constantFeatures, wi, treeId, nodeId, LOCAL);
    // Note that auto-binning occurs here (if enabled). If there are more values for
    // a feature than autobinSize, randomly select potential split values with probability:
    // number-of-values / autobinSize.  This stochastically chooses about autobinSize values
    // out of the full set.
    // Note: For efficiency, we use autobinSize * 2**32-1 (i.e. autobinSizeScald) so that we can directly compare to RANDOM()
    //       without having to divide by 2**32-1
    // We also remove any features that are constant (i.e. have only one value) at this node and
    // we eliminate the first value for ordered features, since there are only N-1 splits.  Note:
    // we recognize this case by isOrd AND altVal = value (no midpoint assigned).
    goodFeatures := features(vals > 1);
    featureVals2 := JOIN(featureValsM, goodFeatures, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId
    //featureVals2 := JOIN(featureValsM, features(vals > 1), LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId
                            AND LEFT.nodeId = RIGHT.nodeId AND LEFT.number = RIGHT.number,
                         TRANSFORM({featureVals},
                         SELF.altVal := IF(LEFT.isOrdinal AND LEFT.altVal = LEFT.value OR
                         //SELF.altVal := IF(LEFT.isOrdinal AND LEFT.altVal = -99999999 OR
                                        (autoBin = TRUE AND RIGHT.vals >= autobinSize AND
                                        RANDOM() > autobinSizeScald / RIGHT.vals),
                                        SKIP, LEFT.altVal),
                         SELF := LEFT), LOCAL);
    // Replicate each datapoint for the node to every possible split for that node
    // Mark each datapoint as being left or right of the split.  Handle both Ordinal and Nominal cases.
    allSplitDat := JOIN(nodeDat, featureVals2, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId
                        AND RIGHT.nodeId = LEFT.nodeId AND LEFT.number = RIGHT.number,
                      TRANSFORM({TreeNodeDat, t_FieldReal splitVal},
                                SELF.splitVal := RIGHT.altVal,
                                SELF.isLEFT := IF((LEFT.isOrdinal AND LEFT.value <= SELF.splitVal)
                                      OR (NOT LEFT.isOrdinal AND LEFT.value = SELF.splitVal),TRUE, FALSE),
                                SELF := LEFT), LOCAL);
    // Calculate the variance of the left and right groups of each split
    // Calculate the variance for left and right splits.  Note that parentId is constant for a node,
    // but we need to pull it through so we have it in the output
    lrVariances := TABLE(allSplitDat, {wi, treeId, nodeId, number, splitVal, isLeft, parentId, isOrdinal,
                            UNSIGNED cnt := COUNT(GROUP), REAL var := VARIANCE(GROUP, depend)},
                            wi, treeId, nodeId, number, splitVal, isLeft, parentId, isOrdinal, LOCAL);
    // Now sum the variance * count for left and right
    splitVariance := TABLE(lrVariances, {wi, treeId, nodeId, number, splitVal, parentId, isOrdinal,
                                REAL splitVariance := SUM(GROUP, var*cnt) / SUM(GROUP,cnt)},
                              wi, treeId, nodeId, number, splitVal, parentId, isOrdinal, LOCAL);
    // Use Variance Reduction (VR) to find the best split for each node (i.e. the one with the greatest VR)
    // Note that there is a chance that we may have eliminated all of the features e.g., because all of the
    // selected features for this node were constant.  This is why we do a RIGHT OUTER JOIN to make sure
    // that we have at least one answer per parent node.
    // We detect this case by number = 0 (null value).  In this case, we throw in a split on feature number
    // maxU4 (maximum value of Unsigned 4), which will not make any progress, but will keep the ball rolling. Note that
    // number = 0 would indicate no progress and therefore cause the node to be treated as a mixed leaf.
    // For Random Forest, we want to keep going because we might find a better feature in the next round.
    vr := JOIN(splitVariance, parentSummary, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId AND
                                              LEFT.nodeId = RIGHT.nodeId,
                TRANSFORM({splitVariance, BOOLEAN isLeft, REAL8 ir, t_RecordId support},
                          SELF.splitVariance := LEFT.splitVariance,
                          SELF.ir := IF(LEFT.number = 0, 0, RIGHT.impurity - LEFT.splitVariance),
                          SELF.number := IF(LEFT.number = 0 AND allowNoProgress, maxU4,
                                            IF(SELF.ir > minVarRed OR allowNoProgress, LEFT.number, 0)),
                          SELF.splitVal := LEFT.splitVal,
                          SELF.isOrdinal := IF(LEFT.number = 0, FALSE, LEFT.isOrdinal),
                          SELF := RIGHT), RIGHT OUTER, LOCAL);
                          //SELF := RIGHT), RIGHT OUTER, LOCAL);
    // Choose the split with the greatest IR (i.e. Impurity Reduction = Variance Reduction)
    vrS := SORT(vr, wi, treeId, nodeId, -ir, LOCAL);
    bestSplits0 := DEDUP(vrS, wi, treeId, nodeId, LOCAL);
    // In the case where we had no non-constant splits we fill in with an arbitrary one of the constant
    // splits so that we can keep the tree growing with the next set of selected features on the next
    // round.  Otherwise, the tree would be truncated.
    bestSplits := JOIN(bestSplits0, dummySplits, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId AND
                         LEFT.nodeId = RIGHT.nodeId,
                         TRANSFORM(SplitDat,
                            SELF.number := IF(LEFT.number = maxU4, RIGHT.number, LEFT.number),
                            SELF.splitVal := IF(LEFT.number = maxU4, maxR8, LEFT.splitVal), // Force LEFT
                            SELF.isOrdinal := IF(LEFT.number = maxU4, TRUE, LEFT.isOrdinal), // Always treat as ordered
                            SELF := LEFT), LEFT OUTER, LOCAL);
    RETURN bestSplits;
  END; // FindBestSplit

  // Grow one layer of the forest
  SHARED DATASET(TreeNodeDat) GrowForestLevel(DATASET(TreeNodeDat) nodeDat, t_Count treeLevel) := FUNCTION
    // Calculate the Impurity for each node.
    //nodeDatS := SORT(nodeDat, wi, treeId, nodeId, LOCAL);
    // NodeSummary has one record per node
    nodeSummary := TABLE(nodeDat, {wi, treeId, nodeId, parentId, isLeft, var:= VARIANCE(GROUP, depend),
                                    cnt := COUNT(GROUP), mean := AVE(GROUP, depend)},
                            wi, treeId, nodeId, parentId, isLeft, LOCAL);
    nodeImp := PROJECT(nodeSummary, TRANSFORM(NodeImpurity, SELF.impurity := LEFT.var,
                                              SELF.support := LEFT.cnt, SELF := LEFT));

    // Filtering pure and non-pure nodes. We translate any pure nodes and their associated data into a leaf node.
    // Impure nodes need further splitting, so they are passed into the next phase.
    // If we're at the maxDepth of the tree, then consider all nodes as pure enough, since they
    // won't get any purer.
    pureEnoughNodes := nodeSummary(var < PureNodeThreshold OR treeLevel = maxDepth);

    // Eliminate any data associated with the leafNodes from the original node data.  What's left
    // is the data for the impure nodes that still need to be split
    toSplitNodes := JOIN(nodeSummary, pureEnoughNodes, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId AND
                      LEFT.nodeId = RIGHT.nodeId,
                    TRANSFORM(TreeNodeDat, SELF := LEFT, SELF := []),
                    LEFT ONLY, LOCAL);
    // Choose a random set of feature on which to split each node, and extend the data with
    // the values of each of those features (X)
    toSplitVars := SelectVarsForNodes(toSplitNodes);
    // Now, extend the values of each of those features (X) for each id
    // Use the indices to get the corresponding X value for each field.
    // Redistribute by id to match up with the original X data
    toSplitDat0 := JOIN(toSplitVars, nodeDat, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId AND
                              LEFT.nodeId = RIGHT.nodeId, TRANSFORM(TreeNodeDat, SELF.number := LEFT.number,
                              SELF := RIGHT), LEFT OUTER, LOCAL);
    // Redistribute by id to match up with the original X data, and sort to align the JOIN.
    toSplitDat1:= SORT(DISTRIBUTE(toSplitDat0, HASH32(wi, origId)), wi, origId, number, LOCAL);
    toSplitDat2 := JOIN(toSplitDat1, X, LEFT.wi = RIGHT.wi AND LEFT.origId=RIGHT.id AND LEFT.number=RIGHT.number,
                        TRANSFORM(TreeNodeDat, SELF.value := RIGHT.value, SELF.isOrdinal := RIGHT.isOrdinal, SELF := LEFT),
                        LOCAL);
    // Now redistribute the results by treeId for further analysis. Restore the original sort order.
    toSplitDat := DISTRIBUTE(toSplitDat2, HASH32(wi, treeId));

    // Filter nodeImp so that only the "not pure enough" nodes are included.  This is important
    // because we use this as the set of nodes for which we need to find best splits.
    parentNodeImp := nodeImp(impurity >= PureNodeThreshold AND treeLevel < maxDepth);

    // Now try all the possible splits and find the best
    bestSplits := findBestSplit(toSplitDat, parentNodeImp);

    // Reasonable splits were found
    goodSplits := bestSplits(number != 0);
    // No split made any progress, or we are at maxDepth for the tree
    badSplits := bestSplits(number = 0);
    badSplitNodes := JOIN(badSplits, nodeSummary, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId AND
                              LEFT.nodeId = RIGHT.nodeId, TRANSFORM(RIGHT), LOCAL);
    // Remove from toSplitDat any cells that are 1) from a bad split or 2) for a feature that was
    // not chosen as the best split. Call it goodSplitDat.
    goodSplitDat := JOIN(toSplitDat, goodSplits, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId AND
                            LEFT.nodeId = RIGHT.nodeId AND LEFT.number = RIGHT.number, TRANSFORM(LEFT), LOCAL);
    // Now, create a split node and two child nodes for each split.
    // First move the data to new child nodes.
    // Start by finding the data samples that fit into the left and the right

    leftIds := JOIN(goodSplits, goodSplitDat, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId AND
                        LEFT.nodeId = RIGHT.nodeId AND LEFT.number = RIGHT.number AND
                        ((RIGHT.isOrdinal AND RIGHT.value <= LEFT.splitVal) OR
                          (NOT RIGHT.isOrdinal AND RIGHT.value = LEFT.splitVal)),
                      TRANSFORM({t_Work_Item wi, t_TreeId treeId, t_NodeId nodeId, t_RecordId id},
                        SELF.treeId := LEFT.treeId, SELF.nodeId := LEFT.nodeId, SELF.id := RIGHT.id,
                        SELF.wi := RIGHT.wi),
                      LEFT OUTER, LOCAL);
    // Assign the data ids to either the left or right branch at the next level
    // All of the node data for the left split (i.e. for Ordinal data: where val <= splitVal,
    //  for Nominal data: where val = splitVal) is marked LEFT.
    // All the node data for the right split(i.e. for Ordinal data: where val > splitVal,
    //  for Nominal data: where val <> splitVal) is marked NOT LEFT
    // Note that nodeIds only need to be unique within a level.
    // Left ids are assigned every other value (1, 3, 5, ...) to leave room for the rights,
    // which will be left plus 1 for a given parent node.  This provides an inexpensive way to assign
    // ids at the next level (though it opens the door for overflow of nodeId).  We handle that
    // case later.
    // Note that 'number' is set to zero for next level data.  New features will be selected next time around.
    LR_nextLevel := JOIN(goodSplitDat, leftIds, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId AND
                      LEFT.nodeId = RIGHT.nodeId AND LEFT.id = RIGHT.id,
                      TRANSFORM(TreeNodeDat, SELF.level := treeLevel + 1,
                                SELF.nodeId := IF(RIGHT.treeId > 0, LEFT.nodeId * 2 - 1, LEFT.nodeId * 2),
                                SELF.parentId := LEFT.nodeId,
                                SELF.isLeft := IF(RIGHT.treeId > 0, TRUE, FALSE),
                                SELF.number := 0;
                                SELF := LEFT), LEFT OUTER, LOCAL);
    // Occasionally, recalculate the nodeIds to make them contiguous to avoid an overflow
    // error when the trees get very deep.  Note that nodeId only needs to be unique within
    // a level.  It is not required that they be a function of the parent's id since parentId will
    // anchor the child to its parent.
    nextLevelIds := TABLE(LR_nextLevel, {wi, treeId, nodeId, t_NodeID newId := 0}, wi, treeId, nodeId, LOCAL);
    nextLevelIdsG := GROUP(nextLevelIds, wi, treeId, LOCAL);
    newIdsG := PROJECT(nextLevelIdsG, TRANSFORM({nextLevelIds}, SELF.newId := COUNTER, SELF := LEFT));
    newIds := UNGROUP(newIdsG);
    fixupIds := SORT(JOIN(LR_nextLevel, newIds, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId AND
                          LEFT.nodeId = RIGHT.nodeId,
                      TRANSFORM(TreeNodeDat, SELF.nodeId := RIGHT.newId, SELF := LEFT), LOCAL), wi, treeId, nodeId, LOCAL);
    maxNodeId := MAX(LR_nextLevel, nodeId);
    // 2**48 is the optimum wrap point.  It allows us to reorganize as infrequently as possible, yet will fit into
    // a Layout_Model2 field.
    nextLevelDat := IF(maxNodeId >= POWER(2, 48), fixupIds, LR_nextLevel);
    // Now reduce each splitNode to a single skeleton node with no data.
    // For a split node (i.e. branch), we only use treeId, nodeId, number (the field number to split on),
    // value (the value to split on), support (# of data records), ir (impurity reduction), and parent-id
    splitNodes := PROJECT(goodSplits, TRANSFORM(TreeNodeDat, SELF.level := treeLevel, SELF.wi := LEFT.wi,
                          SELF.treeId := LEFT.treeId,
                          SELF.nodeId := LEFT.nodeId,
                          self.number := LEFT.number,
                          self.value := LEFT.splitVal,
                          SELF.parentId := LEFT.parentId,
                          SELF.isLeft := LEFT.isLeft,
                          SELF.isOrdinal := LEFT.isOrdinal,
                          SELF.support := LEFT.support,
                          SELF.ir := SQRT(LEFT.ir), // Use square root to reduce the scale.
                          SELF := []));
    // Now handle the leaf nodes, which are the pure-enough nodes, plus the bad splits (i.e. no good
    // split left).
    leafNodes := PROJECT(pureEnoughNodes + badSplitNodes,
                      TRANSFORM(TreeNodeDat, SELF.wi := LEFT.wi, SELF.level := treeLevel,
                              SELF.treeId := LEFT.treeId, SELF.nodeId := LEFT.nodeId,
                              SELF.parentId := LEFT.parentId, SELF.isLeft := LEFT.isLeft, SELF.id := 0, SELF.number := 0,
                              SELF.depend := LEFT.mean, SELF.support := LEFT.cnt, SELF := []), LOCAL);
    // Return the three types of nodes: leafs at this level, splits (branches) at this level, and nodes at the next level (children of the branches).
    RETURN leafNodes + splitNodes + nextLevelDat;
  END;  // GrowForestLevel

  // Find the corresponding leaf node for each X sample given an expanded forest model (set of tree nodes)
  EXPORT DATASET(TreeNodeDat) GetLeafsForDatax(DATASET(TreeNodeDat) tNodes, DATASET(GenField) X) := FUNCTION
    // Distribute X by wi and id.
    x_D := DISTRIBUTE(X, HASH32(wi, id));
    // Extend each root for each ID in X
    // Leave the extended roots distributed by wi, id.
    roots := tNodes(level = 1);
    allIds := DEDUP(x_D, wi, id, LOCAL);
    rootsExt := JOIN(allIds, roots, LEFT.wi = RIGHT.wi, TRANSFORM(TreeNodeDat, SELF.id := LEFT.id, SELF := RIGHT),
                     MANY, LOOKUP);
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

  // Produce a prediction for each X sample given an expanded forest model (set of tree nodes)
  EXPORT DATASET(NumericField) ForestPredict(DATASET(TreeNodeDat) tNodes, DATASET(GenField) X) := FUNCTION
    selectedLeafs := GetLeafsForData(tNodes, X);
    // At this point, we have one leaf node per tree per datapoint (id)
    // The leafs are distributed per wi, id.
    // The leaf nodes contain the final prediction in their 'depend' field.
    // Now we take the average prediction across all trees to get the final prediction.
    results0 := TABLE(selectedLeafs, {wi, id, avg := AVE(GROUP, depend)},
                        wi, id, LOCAL);
    results := PROJECT(results0, TRANSFORM(NumericField, SELF.number := 1, SELF.value := LEFT.avg, SELF := LEFT));
    RETURN results;
  END; // ForestPredict

  // Create a regression forest from the X, Y data, or use model passed in.
  // Use that forest model to predict the Y for a set of X values.
  EXPORT Predict(DATASET(GenField) X, DATASET(Layout_Model2) mod=empty_model) := FUNCTION
    tNodes := IF(EXISTS(mod), Model2Nodes(mod), GetNodes);
    predictions := ForestPredict(tNodes, X);
    RETURN predictions;
  END;

  // Calculate R-squared, the Coefficient of Determination for the regression.
  EXPORT Rsquared(DATASET(GenField) X, DATASET(GenField) Y, DATASET(Layout_Model2) mod) := FUNCTION
    meanY := AVE(Y, value);
    Yhat := Predict(X, mod);
    sqE := JOIN(Y, Yhat, LEFT.wi = RIGHT.wi AND LEFT.id = RIGHT.id,
                   TRANSFORM({UNSIGNED wi, UNSIGNED id, REAL tss, REAL rss},
                     SELF.tss := POWER(LEFT.value - meanY, 2), // Total Sum-of-squares
                     SELF.rss := POWER(LEFT.value - RIGHT.value, 2), // Residual SS
                     SELF := LEFT));
    result := TABLE(sqE, {wi, R2 := 1 - SUM(GROUP, rss) / SUM(GROUP, tss)}, wi);
    return result;
  END;
END; // RF_Regression (Regression Forest)