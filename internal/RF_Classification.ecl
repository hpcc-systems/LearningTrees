IMPORT $.^ AS LT;
IMPORT LT.internal AS int;
IMPORT LT.LT_Types AS Types;
IMPORT ML_Core as ML;
IMPORT ML.Types AS CTypes;
IMPORT std.system.Thorlib;
IMPORT ndArray;

GenField := Types.GenField;
TreeNodeDat := Types.TreeNodeDat;
SplitDat := Types.SplitDat;
NodeImpurity := Types.NodeImpurity;
wiInfo := Types.wiInfo;t_Work_Item := CTypes.t_Work_Item;
t_Count := CTypes.t_Count;
t_RecordId := CTypes.t_RecordID;
t_FieldNumber := CTypes.t_FieldNumber;
t_TreeId := t_FieldNumber;
t_FieldReal := CTypes.t_FieldReal;
t_Discrete := CTypes.t_Discrete;
Layout_Model := CTypes.Layout_Model;
t_NodeId := Types.t_NodeId;
DiscreteField := CTypes.DiscreteField;
Layout_Model2 := Types.Layout_Model2;
ClassProbs := Types.ClassProbs;

/**
  * Classification Forest Module
  *
  * This module provides a Random Forest Classifier based on Breiman, 2001
  * with extensions.
  *
  */
EXPORT RF_Classification(DATASET(GenField) X_in=DATASET([], GenField),
                          DATASET(GenField) Y_In=DATASET([], GenField),
                          UNSIGNED numTrees=100,
                          UNSIGNED featuresPerNode=0,
                          UNSIGNED maxDepth=255) := MODULE(int.RF_Base(X_in, Y_in, numTrees,
                                                            featuresPerNode, maxDepth))
  SHARED allowNoProgress := TRUE;  // If FALSE, tree will terminate when no progess can be made on any
                                   // feature.  For RF, should be TRUE since it may get a better choice
                                   // of features at the next level.
  SHARED minImpurity := .0000001;   // Nodes with impurity less than this are considered pure.
  SHARED classWeightsRec := RECORD
    t_work_item wi;
    t_Discrete classLabel;
    t_FieldReal weight;
  END;
  SHARED classWeights := FUNCTION
    minClassWeight := .25; // Offset so that no weight can approach zero.
    // The weight of each class is assigned as a logarithmic inverse of the class frequency
    Y_DS := DISTRIBUTE(Y_in, HASH32(wi, value));
    classCounts := TABLE(Y_DS, {wi, value, cnt := COUNT(GROUP)}, wi, value);
    // Calculate the weights as classWeight(class) := -LN(<proportion of records of class>) + minClassWeight
    classWeights := JOIN(classCounts, wiMeta, LEFT.wi = RIGHT.wi, TRANSFORM(classWeightsRec,
                                          SELF.wi := LEFT.wi,
                                          SELF.classLabel := LEFT.value,
                                          SELF.weight := -LN(LEFT.cnt / RIGHT.numSamples) + minClassWeight), LOOKUP);
    return classWeights;
  END;
  // Find the best split for a given set of nodes.  In this case, it is the one with the highest information
  // gain.  Every possible split point is considered for each independent variable in the tree.
  // For nominal variables, the split is an equality split on one of the possible values for that variable
  // (i.e. split into = s and != s).  For ordinal variables, the split is an inequality (i.e. split into <= s and > s)
  // For each node, the split with the highest Information Gain (IG) is returned.
  SHARED DATASET(SplitDat) findBestSplit(DATASET(TreeNodeDat) nodeVarDat, DATASET(NodeImpurity) parentEntropy) := FUNCTION
    // Calculate the Information Gain (IG) for each split.
    // IG := Entropy(H) of Parent - Entropy(H) of the proposed split := H-parent - SUM(prob(child) * H-child) for each child group of the split
    // IV := -SUM(Prob(x) * Log2(Prob(x)) for all values of X independent variable
    // H := -SUM(Prob(y) * Log2(Prob(y)) for all values of Y dependent variable
    // At this point, nodeVarDat has one record per node per selected feature per id
    // Start by getting a list of all the values for each feature per node
    featureVals := TABLE(nodeVarDat, {wi, treeId, nodeId, number, value, isOrdinal,
                            cnt := COUNT(GROUP)},
                          wi, treeId, nodeId, number, value, isOrdinal, LOCAL);
    // Calculate the number of values per feature per node
    features0 := TABLE(featureVals, {wi, treeId, nodeId, number, tot := SUM(GROUP, cnt), gmax := MAX(GROUP, value),
                            vals := COUNT(GROUP)},
                          wi, treeId, nodeId, number, LOCAL);
    //features := features0(vals > 1); // Eliminate any features with constant value for the node
    features := features0;
    // Note that auto-binning occurs here (if enabled). If there are more values for
    // a feature than autobinSize, randomly select potential split values with probability:
    // 1/(number-of-values / autobinSize).
    // Note: For efficiency, we use autobinSize * 2**32-1 so that we can directly compare to RANDOM()
    //       without having to divide by 2**32-1
    featureVals2 := JOIN(featureVals, features, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId
                            AND LEFT.nodeId = RIGHT.nodeId AND LEFT.number = RIGHT.number,
                          TRANSFORM({featureVals, REAL prop, REAL plogp, t_FieldReal gmax, UNSIGNED tot},
                                      SELF.prop := LEFT.cnt / RIGHT.tot,
                                      SELF.plogp := P_Log_P(SELF.prop),
                                      SELF.gmax := RIGHT.gmax,
                                      SELF.tot := IF(autoBin = FALSE OR
                                        RIGHT.vals < autobinSize OR
                                        RANDOM() < autobinSizeScald/RIGHT.vals,
                                        RIGHT.tot, SKIP),
                                      SELF := LEFT), LOCAL);
    // Filter the feature values so that we don't replicate data for the last data-point, except
    // for nominal features with more than two values.  This is strictly an optimization, since for
    // binary nominals, splitting on one value is the same as splitting on the other, and for ordinals,
    // if the last value is used, it will not result in any information gain since all data will be
    // to the left of the split.
    featureVals3 := featureVals2((NOT isOrdinal AND tot > 2) OR value != gmax);
    // Replicate each datapoint for the node to every possible split for that node
    // Mark each datapoint as being left or right of the split.  Handle both Ordinal and Nominal cases.
    allSplitDat := JOIN(nodeVarDat, featureVals3, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId
                        AND RIGHT.nodeId = LEFT.nodeId AND LEFT.number = RIGHT.number,
                      TRANSFORM({TreeNodeDat, t_FieldReal splitVal}, SELF.splitVal := RIGHT.value,
                                SELF.isLEFT := IF((LEFT.isOrdinal AND LEFT.value <= SELF.splitVal)
                                                  OR (NOT LEFT.isOrdinal AND LEFT.value = SELF.splitVal),TRUE, FALSE),
                                SELF := LEFT), LEFT OUTER, LOCAL);
    // Calculate the entropy of the left and right groups of each split
    // Group by value of Y (depend) for left and right splits
    dependGroups := TABLE(allSplitDat, {wi, treeId, nodeId, number, splitVal, isLeft, depend,
                              isOrdinal, UNSIGNED cnt := COUNT(GROUP)},
                            wi, treeId, nodeId, number, splitVal, isLeft, depend, isOrdinal, LOCAL);
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
    lr_entropies := TABLE(dependRatios, {wi, treeId, nodeId, number, splitVal, isLeft, isOrdinal, tot := SUM(GROUP, cnt),
                            entropy := SUM(GROUP, plogp)},
                          wi, treeId, nodeId, number, splitVal, isLeft, isOrdinal, LOCAL);
    // Now calculate the weighted average of entropies of the two groups (weighted by number of datapoints in each)
    // Note that 'tot' is number of datapoints for each side of the split.
    entropies0 := TABLE(lr_entropies, {wi, treeId, nodeId, number, splitVal, isOrdinal,
                               REAL totEntropy := SUM(GROUP, entropy * tot) / SUM(GROUP, tot)},
                               //REAL totEntropy := SUM(GROUP, entropy)},
                              wi, treeId, nodeId, number, splitVal, isOrdinal, LOCAL);
    entropies := SORT(entropies0, wi, treeId, nodeId, totEntropy, LOCAL);
    // We only care about the split with the lowest entropy for each tree node.  Since the parentEntropy
    // is constant for a given tree node, the split with the lowest entropy will also be the split
    // with the highest Information Gain.
    lowestEntropies := DEDUP(entropies, wi, treeId, nodeId, LOCAL);
    // Now calculate Information Gain
    // In order to stop the tree-building process when there is no split that gives information-gain
    // we set 'number' to zero to indicate that there is no best split when we hit that case.
    // That happens when the data is not fully separable by the independent variables.
    ig := JOIN(lowestEntropies, parentEntropy, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId AND
                  LEFT.nodeId = RIGHT.nodeId,
                TRANSFORM({entropies, t_NodeID parentId, BOOLEAN isLeft, REAL ig},
                          SELF.ig := RIGHT.impurity - LEFT.totEntropy,
                          SELF.number := IF(SELF.ig > 0 OR allowNoProgress, LEFT.number, 0),
                          SELF.parentId := RIGHT.parentId, SELF.isLeft := RIGHT.isLeft, SELF := LEFT),
                LOCAL);
    // Choose the split with the greatest information gain for each node
    bestSplits := ig;
    RETURN PROJECT(bestSplits, SplitDat);
  END;

  // Grow one layer of the forest
  SHARED DATASET(TreeNodeDat) GrowForestLevel(DATASET(TreeNodeDat) nodeDat, t_Count treeLevel) := FUNCTION
    // At this point, nodes contains one element per wi, treeId, nodeId and id within the node.
    // The number field is not used at this point, nor is the value field.  The depend field has
    // the dependent value (Y) for each id.
    // Calculate the Impurity for each node.
    // NodeValCounts has one record per node, per value of the dependent variable (Y)
    nodeValCounts := TABLE(nodeDat, {wi, treeId, nodeId, depend, parentId, isLeft, cnt:= COUNT(GROUP)},
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
    // Node impurity
    nodeImp := PROJECT(nodeEnt0, TRANSFORM(NodeImpurity, SELF.impurity := LEFT.entropy, SELF := LEFT));

    // Filtering pure and non-pure nodes. We translate any pure nodes and their associated data into a leaf node.
    // Impure nodes need further splitting, so they are passed into the next phase.
    // If we are at maxDepth, consider everything pure enough.
    pureEnoughNodes := nodeImp(impurity < minImpurity OR treeLevel = maxDepth);  // Nodes considered pure enough.

    // Eliminate any data associated with the leafNodes from the original node data.  What's left
    // is the data for the impure nodes that still need to be split
    toSplitNodes := JOIN(nodeCounts, pureEnoughNodes, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId AND
                        LEFT.nodeId = RIGHT.nodeId,
                      TRANSFORM(TreeNodeDat, SELF := LEFT, SELF := []),
                      LEFT ONLY, LOCAL);
    // Choose a random set of feature on which to split each node
    // At this point, we have one record per tree, node, and number (for selected features)
    toSplitVars := SelectVarsForNodes(toSplitNodes);

    // Now, extend the values of each of those features (X) for each id
    // Use the indices to get the corresponding X value for each field.
    // Redistribute by id to match up with the original X data
    toSplitDat0 := JOIN(toSplitVars, nodeDat, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId AND
                              LEFT.nodeId = RIGHT.nodeId, TRANSFORM(TreeNodeDat, SELF.number := LEFT.number,
                              SELF := RIGHT), LOCAL);
    // Redistribute by id to match up with the original X data, and sort to align the JOIN.
    toSplitDat1:= SORT(DISTRIBUTE(toSplitDat0, HASH32(wi, origId)), wi, origId, number, LOCAL);
    toSplitDat2 := JOIN(toSplitDat1, X, LEFT.wi = RIGHT.wi AND LEFT.origId=RIGHT.id AND LEFT.number=RIGHT.number,
                        TRANSFORM(TreeNodeDat, SELF.value := RIGHT.value, SELF.isOrdinal := RIGHT.isOrdinal, SELF := LEFT),
                        LOCAL);
    // Now redistribute the results by treeId for further analysis.  Sort for further analysis.
    toSplitDat := DISTRIBUTE(toSplitDat2, HASH32(wi, treeId));
//    toSplitDat := SORT(DISTRIBUTE(toSplitDat2, HASH32(wi, treeId)), wi, treeId, nodeId, LOCAL);

    // Now try all the possible splits and find the best
    bestSplits := findBestSplit(toSplitDat, nodeImp);
    // Reasonable splits were found
    goodSplits := bestSplits(number != 0);
    // No split made any progress, or we are at maxDepth for the tree
    badSplits := bestSplits(number = 0);

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
                      LOCAL);
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
    nextLevelDat := IF(treeLevel % 32 = 0, fixupIds, LR_nextLevel); // Recalculate every 32 levels to avoid overflow
    // Now reduce each splitNode to a single skeleton node with no data.
    // For a split node (i.e. branch), we only use treeId, nodeId, number (the field number to split on), value (the value to split on), and parent-id
    splitNodes := PROJECT(goodSplits, TRANSFORM(TreeNodeDat, SELF.level := treeLevel, SELF.wi := LEFT.wi,
                          SELF.treeId := LEFT.treeId,
                          SELF.nodeId := LEFT.nodeId, self.number := LEFT.number, self.value := LEFT.splitVal,
                          SELF.isOrdinal := LEFT.isOrdinal,
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
    pureNodes := JOIN(nodeValCounts, pureEnoughNodes, LEFT.wi = RIGHT.wi AND LEFT.treeId = RIGHT.treeId AND
                        LEFT.nodeId = RIGHT.nodeId,
                      TRANSFORM(TreeNodeDat, SELF.wi := LEFT.wi, SELF.level := treeLevel,
                              SELF.treeId := LEFT.treeId, SELF.nodeId := LEFT.nodeId,
                              SELF.parentId := LEFT.parentId,
                              SELF.isLeft := LEFT.isLeft, SELF.id := 0, SELF.number := 0,
                              SELF.depend := LEFT.depend, SELF.support := LEFT.cnt, SELF := []), LOCAL);
    leafNodes := pureNodes + mixedLeafs;
    // Return the three types of nodes: leafs at this level, splits (branches) at this level, and nodes at
    // the next level (children of the branches).
    RETURN leafNodes + splitNodes + nextLevelDat;
  END;

  SHARED emptyClassWeights := DATASET([], classWeightsRec);
  // Get the probability of each sample belonging to each class,
  // given an expanded forest model (set of tree nodes)
  // Note that 'probability' is used loosely here as a percentage
  // of trees that voted for each class.
  SHARED DATASET(ClassProbs) FClassProbabilities(DATASET(TreeNodeDat) tNodes, DATASET(GenField) X,
                                                  DATASET(classWeightsRec) classWts=emptyClassWeights) := FUNCTION
    modTreeCount := MAX(tNodes, treeId);  // Number of trees in the model
    // Distribute X by wi and id.
    xD := DISTRIBUTE(X, HASH32(wi, id));
    // Extend each root for each ID in X
    roots := tNodes(level = 1);
    rootsExt := JOIN(xD(number=1), roots, LEFT.wi = RIGHT.wi, TRANSFORM(TreeNodeDat, SELF.id := LEFT.id, SELF := RIGHT),
                     MANY, LOOKUP);
    // RootsExt is now distributed by wi, id.
    rootBranches := rootsExt(number != 0); // Roots are almost always branch (split) nodes.
    rootLeafs := rootsExt(number = 0); // Unusual but not impossible
    loopBody(DATASET(TreeNodeDat) levelBranches, UNSIGNED tLevel) := FUNCTION
      // At this point, we have one record per node, per id.
      // We extend each id down the tree one level at a time, picking the correct next nodes
      // for that id at each branch.
      // Next nodes are returned -- both leafs and branches.  The leafs are filtered out by the LOOP,
      // while the branches are send on to the next round.
      // Ultimately, a leaf is returned for each id and tree, which defines our final result.
      // Select the next nodes by combining the selected data field with each node
      // Note that:  1) we retain the id from the previous round, but the field number(number) is derived from the branch
      //             2) 'value' in the node is the value to split upon, while value in the data (X) is the value of that
      //                  datapoint
      branchVals := JOIN(levelBranches, xD, LEFT.wi = RIGHT.wi AND LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number,
                          TRANSFORM({TreeNodeDat, BOOLEAN branchLeft},
                                      SELF.branchLeft :=  ((LEFT.isOrdinal AND RIGHT.value <= LEFT.value) OR
                                                          ((NOT LEFT.isOrdinal) AND RIGHT.value = LEFT.value)),
                                      SELF.parentId := LEFT.nodeId, SELF := LEFT),
                          LOCAL);
      // BranchLeft is now true for all records that need to go down the left branch and false for those on the
      // right branch.
      nextLevelNodes := tNodes(level = tLevel + 1);
      // Use LOOKUP JOIN so that nextLevelNodes gets replicated to every node where it is needed.
      // NextLevelSelNodes is left distributed by wi, id.
      nextLevelSelNodes := JOIN(branchVals, nextLevelNodes, LEFT.wi = RIGHT.wi AND
                              LEFT.treeId = RIGHT.treeId AND LEFT.nodeId = RIGHT.parentId AND
                              LEFT.branchLeft = RIGHT.isLeft,
                              TRANSFORM(TreeNodeDat, SELF.id := LEFT.id, SELF := RIGHT), LOOKUP);
      // Return the selected nodes at the next level.  These nodes may be leafs or branches.
      // Any leafs will be filtered out by the loop.  Any branches will go on to the next round.
      // When there are no more branches to process, we are done.  The selected leafs for each datapoint
      // is returned.
      RETURN nextLevelSelNodes;
    END; // Loop Body
    // The loop will return the leaf node associated with each sample for each tree.
    selectedLeafs0 := LOOP(rootBranches, LEFT.number>0, EXISTS(ROWS(LEFT)),
                          loopBody(ROWS(LEFT), COUNTER));
    selectedLeafs := selectedLeafs0 + rootLeafs;
    // At this point, we have one leaf node per tree per datapoint (X)
    // The leaf nodes contain the final class in their 'depend' field.
    // Now we need to count the votes for each class and id
    // Calculate raw (unweighted) probabilities
    probs0 := TABLE(selectedLeafs, {wi, id, depend, cnt := COUNT(GROUP), prob := COUNT(GROUP) / modTreeCount},
                    wi, id, depend, LOCAL);
    // Function to calculate weighted probabilities.
    calcWeightedProbs := FUNCTION
      // Calculate prob * weight for each id
      wprobs0 := JOIN(probs0, classWts, LEFT.wi = RIGHT.wi AND
                          LEFT.depend = RIGHT.classLabel,
                        TRANSFORM({probs0}, SELF.prob := LEFT.prob * RIGHT.weight,
                                        SELF := LEFT), LOOKUP);
      // Normalize based on the sum of weighted probabilities for each id
      totWprobs := TABLE(wprobs0, {wi, id, tot := SUM(GROUP, prob)}, wi, id, LOCAL);
      wprobs := JOIN(wprobs0, totWprobs, LEFT.wi = RIGHT.wi AND LEFT.id = RIGHT.id,
                        TRANSFORM({wprobs0}, SELF.prob := LEFT.prob / RIGHT.tot, SELF := LEFT), LOCAL);
      // Return prob = raw_prob * weight / SUM(raw_prob * weight) for each id
      return wprobs;
    END; // calcWeightedProbs
    // If weights were provided, use weighted probs, otherwise raw probs
    wprobs := IF(EXISTS(classWts), calcWeightedProbs, probs0);
    // Now one record per datapoint per value of depend (Y) with the count of 'votes' and
    // proportion of votes (raw prob) or class weighted proportion for each depend value.
    RETURN PROJECT(wprobs, TRANSFORM(ClassProbs, SELF.class := LEFT.depend, SELF := LEFT));
  END; // FClassProbabilities

  // Produce a class for each X sample given an expanded forest model (set of tree nodes)
  SHARED DATASET(DiscreteField) ForestClassify(DATASET(TreeNodeDat) tNodes,
                                        DATASET(GenField) X,
                                        DATASET(classWeightsRec) classWts=emptyClassWeights) := FUNCTION
    // Get the probabilities of each sample
    probs := FClassProbabilities(tNodes, X, classWts);
    // Reduce to one record per datapoint, with the highest class probability winning
    probsS := SORT(probs, wi, id, -prob, LOCAL);
    // Keep the first leaf value for each wi and id.  That is the one with the highest probability
    selectedClasses := DEDUP(probsS, wi, id, LOCAL);
    // Transform to discrete field
    results := PROJECT(selectedClasses, TRANSFORM(DiscreteField, SELF.number := 1, SELF.value := LEFT.class, SELF := LEFT));
    RETURN results;
  END;
  /**
    * Extract the class weights dataset from the model
    *
    */
  EXPORT Model2ClassWeights(DATASET(Layout_Model2) mod) := FUNCTION
    modCW := ndArray.Extract(mod, [Types.rfModInd1.classWeights]);
    cw := PROJECT(modCW, TRANSFORM(classWeightsRec, SELF.wi := LEFT.wi, SELF.classLabel := LEFT.indexes[1],
                                        SELF.weight := LEFT.value));
    RETURN cw;
  END;

  // Use the supplied forest model to predict the ClassLabel(Y) for a set of X values.
  // Optionally use class balancing to weight the classes inversely proportional to their
  // frequency in the training data.
  EXPORT DATASET(DiscreteField) Classify(DATASET(GenField) X, DATASET(Layout_Model2) mod,
                                            BOOLEAN balanceClasses=FALSE) := FUNCTION
    tNodes := Model2Nodes(mod);
    classWts := Model2ClassWeights(mod);
    classes := IF(balanceClasses, ForestClassify(tNodes, X, classWts), ForestClassify(tNodes, X));
    RETURN classes;
  END;

  // Get Class Probabilities.
  // Note that probabilities here are the (optionally class weighted) proportion of trees that
  // 'voted' for each class, for each X sample.
  EXPORT DATASET(ClassProbs) GetClassProbs(DATASET(GenField) X, DATASET(Layout_Model2) mod,
                                            BOOLEAN balanceClasses=FALSE) := FUNCTION
    tNodes := Model2Nodes(mod);
    classWts := Model2ClassWeights(mod);
    probs := IF(balanceClasses, FClassProbabilities(tNodes, X, classWts),FClassProbabilities(tNodes, X));
    RETURN probs;
  END;
  /**
    * Get forest model
    *
    * Overlays the GetModel function of RF Base to provide additional information
    * used only for classification.
    * Adds the class weights, which are only used for classification
    *
    * RF uses the Layout_Model2 format, which is implemented as an N-Dimensional
    * numeric array (i.e. ndArray.NumericArray).
    *
    * See LT_Types for the format of the model
    *
    */
  EXPORT DATASET(Layout_Model2) GetModel := FUNCTION
    //baseMod := int.RF_Base.GetModel;
    nodes := GetNodes;
    mod1 := Nodes2Model(nodes);
    mod2 := Indexes2Model;
    baseMod := mod1 + mod2;
    naClassWeights := PROJECT(classWeights, TRANSFORM(ndArray.NumericArray, SELF.wi := LEFT.wi,
                                                        SELF.indexes := [Types.rfModInd1.classWeights, LEFT.classLabel],
                                                        SELF.value := LEFT.weight));
    mod := baseMod + naClassWeights;
    RETURN mod;
  END;
END; // RF_Classification