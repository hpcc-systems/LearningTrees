IMPORT $.^ AS LT;
IMPORT LT.LT_Types;
IMPORT ML_Core;
IMPORT ML_Core.Types;
IMPORT Std;
IMPORT Std.System.Thorlib;
GenField := LT_Types.GenField;
TreeNodeDat := LT_Types.TreeNodeDat; // Tree nodes for forest
BfTreeNodeDat := LT_Types.BfTreeNodeDat; // Tree nodes for Gradient Boosting
RF_Regression := LT.internal.RF_Regression;
RF_Base := LT.internal.RF_Base;
Layout_Model2 := Types.Layout_Model2;
NumericField := Types.NumericField;
FeatureImportanceRec := LT_Types.FeatureImportanceRec;
ModelOps2 := ML_Core.ModelOps2;
fModels := LT_Types.Bf_Model.Ind1.fModels;
FM := LT_Types.Forest_Model;
FM1 := FM.Ind1;

/**
  * Classification using Boosted Forests.
  *
  * Boosted Forests (BF) are a combination of Gradient Boosted Trees (BfT) and
  * Random Forests (RF).  They utilize Boosting to enhance the
  * accuracy of Random Forests.
  * BFs are formed as a hierarchy of boosted Random Forests.
  *
  * A Boosted Forest with a forest size of 1 is
  * essentially the same as BfT.  While this is supported, it is not recommended.
  * BFs with forest size >= 10 have characteristics superior to both RF and BfT.
  * They generally provide accuracy higher than RFs, and better than or equal that
  * of expertly regularized BfTs.  Yet they require no regularization, work well
  * with default parameters, and are as easy to use as RFs.
  *
  * BFs provide an early-stopping capability.  This allows the number of boosting
  * iterations (i.e. maxLevels) to be specified at a high level (default 999),
  * but only boosts for as many iterations as necessary to maximize accuracy.
  * In normal practice, boosting will stop long before the default maxLevels is
  * reached.
  *
  * <p>BF  provides myriad (i.e. multiple independent work-item) support,
  * and support for both numeric and categorical independent data (i.e. X).
  *
  * @param X_in The independent Data in GenField format.
  * @param Y_in The dependent Data in GenField format.
  * @param maxLevels The maximum number of boosting iterations to perform.  This is
  *         overridden by early stopping, and is primarily a failsafe in case the
  *         data is non-separable.  Default (recommended) is 999.
  * @param forestSize The number of trees to use in each Random Forest level.  The
  *         default (recommended) is zero, which indicates that it should be
  *         automatically determined by the software.
  * @param maxTreeDepth The depth to which trees are grown.  Smaller numbers provide
  *         weaker learners that are needed for BfT purposes.  The default (recommended)
  *         is 20.
  */
EXPORT BF_Base(DATASET(GenField) X_in=DATASET([], GenField),
                          DATASET(GenField) Y_in=DATASET([], GenField),
                          UNSIGNED maxLevels=999,
                          UNSIGNED forestSize=0,
                          UNSIGNED maxTreeDepth=20
                          ) := MODULE
  SHARED clusterNodes := Thorlib.nodes();
  // Forest size should normally be between 10 and 50.  If user specified 0,
  // that indicates automatic choice.  If less than 10 nodes, we will use 10.
  // If between 10 and 50 nodes, use the node count.  If > 50 nodes, limit it
  // to 50.  In this way, we balance RF generalization with performance.
  SHARED forestSizeAdj := IF(forestSize = 0, MIN(MAX(clusterNodes, 10), 50), forestSize);
  // Features per node setting for RF.  If forestSize is 1, all features are used
  // as there is no need to randomize the tree.  For larger forest sizes, we use 0
  // to indicate that RF should automatically determine it.
  SHARED fpn := IF(forestSizeAdj = 1, 999999, 0);

  /**
    * Convert the set of nodes describing the BfF to a Model Format
    *
    */
  EXPORT DATASET(Layout_Model2) Nodes2Model(DATASET(BfTreeNodeDat) nodes) := FUNCTION
    myRF := RF_Base();
    empty_mod := DATASET([], Layout_Model2);
    doOneRF(DATASET(Layout_Model2) model, UNSIGNED c) := FUNCTION
      levNodes := nodes(bfLevel = c);
      levMod := myRF.Nodes2Model(levNodes);
      newModel := ModelOps2.Insert(model, levMod, [fModels]+[c]);
      RETURN newModel;
    END;
    maxLev := MAX(nodes, bfLevel);
    outModel := LOOP(empty_mod, maxLev, doOneRF(ROWS(LEFT), COUNTER));
    RETURN outModel;
  END;
  /**
    * Extract BfTreeNodeDat records from the model.
    *
    * @param mod The Boosted Forest Model.
    * @return Set of Tree Nodes in BfTreeNodeDat format.
    */
  EXPORT DATASET(BfTreeNodeDat) Model2Nodes(DATASET(Layout_Model2) mod) := FUNCTION
    // Filter to only the model records that represent forest nodes.
    mod_S := SORT(mod, indexes);
    nodesRecs := mod_S(indexes[1] = fModels AND indexes[3] = FM1.nodes);
    // Separate the each level of the boosting hierarchy, so that we can get the nodes
    // for each forest from LearningForests.
    GroupedByLevel := GROUP(nodesRecs, wi, indexes[2]);
    workRec := RECORD
      DATASET(TreeNodeDat) nodes;
      UNSIGNED bfLevel;
    END;
    myRF := RF_Base();
    workRec getTreeNodes(Layout_Model2 rec, DATASET(Layout_Model2) children) := TRANSFORM
      children_adj := PROJECT(children, TRANSFORM(RECORDOF(LEFT),
                                  SELF.indexes := LEFT.indexes[3 ..],
                                  SELF := LEFT), LOCAL);
      levNodes := myRF.Model2Nodes(children_adj);
      SELF.nodes := levNodes;
      SELF.bfLevel := rec.indexes[2];
    END;
    treeNodes := ROLLUP(GroupedByLevel, GROUP, getTreeNodes(LEFT, ROWS(LEFT)));
    BfTreeNodes := NORMALIZE(treeNodes, COUNT(LEFT.nodes), TRANSFORM(BfTreeNodeDat,
                                    SELF.bfLevel := LEFT.bfLevel,
                                    SELF := LEFT.nodes[COUNTER]));
    RETURN DISTRIBUTE(BfTreeNodes, HASH32(wi, treeId));
  END;
  /**
    * Return feature-importance information (internal).  Based on BfTreeNodeDat input.
    *
    * Feature importance is an indication of how much each feature contributed to the model
    * across all trees and forests.
    *
    */
  EXPORT FeatureImportanceNodes(DATASET(BfTreeNodeDat) nodes) := FUNCTION
    deepest := MAX(nodes, bfLevel);
    initNodes := DATASET([], FeatureImportanceRec);
    myRF := RF_Base();
    doOneLevel(DATASET(FeatureImportanceRec) inNodes, UNSIGNED c) := FUNCTION
      levNodes := nodes(bfLevel = c);
      outRecs := myRF.FeatureImportanceNodes(levNodes);
      RETURN outRecs;
    END;
    allRecs := LOOP(initNodes, FALSE, COUNTER <= maxLevels,
                      doOneLevel(ROWS(LEFT), COUNTER));
    cumRecs := TABLE(allRecs, {wi, number, avgImportance := AVE(GROUP, importance),
                                totUses := SUM(GROUP, uses)}, wi, number);
    outRecs := PROJECT(cumRecs, TRANSFORM(FeatureImportanceRec,
                  SELF.importance := LEFT.avgImportance,
                  SELF.uses := LEFT.totUses,
                  SELF := LEFT), LOCAL);
    RETURN SORT(outRecs, wi, -importance);
  END;
  /**
    * Return feature-importance information from the model.
    *
    * Feature importance is an indication of how much each feature contributed to the model
    * across all trees and forests.
    *
    */
  EXPORT FeatureImportance(DATASET(Layout_Model2) mod) := FUNCTION
    nodes := Model2Nodes(mod);
    fi := FeatureImportanceNodes(nodes);
    RETURN fi;
  END;
END;