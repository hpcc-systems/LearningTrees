IMPORT $.^ AS LT;
IMPORT LT.LT_Types;
IMPORT ML_Core;
IMPORT ML_Core.Types;
IMPORT Std;
IMPORT Std.System.Thorlib;
IMPORT LT.internal;
BF_Base := internal.BF_Base;
GenField := LT_Types.GenField;
ModelStats := LT_Types.ModelStats;
TreeNodeDat := LT_Types.TreeNodeDat; // Tree nodes for forest
BfTreeNodeDat := LT_Types.BfTreeNodeDat; // Tree nodes for Gradient Boosting
RF_Regression := LT.internal.RF_Regression;
RF_Base := LT.internal.RF_Base;
Layout_Model2 := Types.Layout_Model2;
NumericField := Types.NumericField;
ModelOps2 := ML_Core.ModelOps2;
fModels := LT_Types.Bf_Model.Ind1.fModels;
FM := LT_Types.Forest_Model;
FM1 := FM.Ind1;

/**
  * Regression using Boosted Forests.
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
  * of expertly regularized GBTs.  Yet they require no regularization, work well
  * with default parameters, and are as easy to use as RFs.
  *
  * BFs provide an early-stopping capability.  This allows the number of boosting
  * iterations (i.e. maxLevels) to be specified at a high level (default 999),
  * but only boosts for as many iterations as necessary to maximize accuracy.
  * In normal practice, boosting will stop long before the default maxLevels is
  * reached.
  *
  * While we recommend use of the default training parameters, we do allow override of
  * these parameters in order to attempt further regularization or optimization, or
  * in cases where you must use straight GBTs.
  * <p>Here are some guidelines for setting these parameters:<ul>
  * <li>It is not recommended to use forest sizes between 2 and 9.  A forest size of 10
  * is the minimum to provide effective RF generalization.  Forests in this range
  * behave somewhere between GBT and BF, and will probably require regularization.</li>
  * <li>There are three regularization parameters: maxDepthPerTree, learningRate, and
  * maxLevels (i.e. the number of boosting iterations).  These all interact.  If
  * early stopping is enabled (i.e. earlyStopThreshold > 0), then it is not necessary to
  * regularize maxLevels as it will be automatically determined.  For GBT
  * (i.e. treesPerLevel = 1), it is necessary to regularize at least the other two
  * parameters in order to achieve reasonable results.</li>
  * <li>For forestSize > 9, these
  * parameters have very little effect, though slight gains may be possible in certain
  * circumstances.</li>
  * <li>For GBT (forestSize = 1), smaller sizes of maxTreeDepth (3-10) are recommended,
  * as are low values for learningRate (< .5).</li>
  * <li>For BF (forestSize > 9), moderate values are likely to provide optimal results:
  * maxTreeDepth between 7 and 30 and learningRate between .5 and 1.0 might generate
  * good results.<li></ul>
  *
  * <p>BF Regression provides myriad (i.e. multiple independent work-item) support,
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
  *         weaker learners that are needed for GBT purposes.  The default (recommended)
  *         is 20.
  * @param learningRate The distance along the gradient to procede on each boosting
  *         iteration.  The default (recommended) is 1.0.
  * @param earlyStopThreshold A threshold against the RVR (Residual Variance Ratio) to
  *         enable early stopping.  The default threshold (recommended) is .0001, which
  *         indicates that we will stop when 99.99% of the variance in the original
  *         data has been explained by the model.  Setting this value to zero disables
  *         early stopping (not recommended).
  */
EXPORT BF_Regression(DATASET(GenField) X_in=DATASET([], GenField),
                          DATASET(GenField) Y_in=DATASET([], GenField),
                          UNSIGNED maxLevels=255,
                          UNSIGNED forestSize=0,
                          UNSIGNED maxTreeDepth=20,
                          REAL8 learningRate=1.0,
                          REAL8 earlyStopThreshold=.0001) := MODULE(BF_Base(X_in, Y_in, maxLevels,
                                                            forestSize, maxTreeDepth))
  SHARED learningRateAdj := learningRate;
  SHARED huberThreshold := .9;
  SHARED lossTypes := ENUM(UNSIGNED1, Squared=1, Absolute=2, Huber=3);
  SHARED lossType := lossTypes.Squared;
  // Function to calculate the gradients based on the selected lossType.
  SHARED CalcGradients(DATASET(GenField) residuals) := FUNCTION
    //
    Sign(REAL8 value) := value / ABS(value);
    // Gradient for Squared Loss is residual
    Sq_gradient(REAL8 residual) := FUNCTION
      grad := residual;
      RETURN grad;
    END;
    // Gradient for Absolute Loss is Sign(residual)
    Abs_gradient(REAL8 residual) := FUNCTION
      grad := Sign(residual);
      RETURN grad;
    END;
    // Gradient for Huber Loss is
    Huber_gradient(REAL8 residual) := FUNCTION
      grad := Sign(residual);
      RETURN grad;
    END;

    gradients := PROJECT(residuals,
                          TRANSFORM(RECORDOF(LEFT),
                            SELF.value := learningRateAdj * MAP(
                                  lossType=lossTypes.Squared => Sq_Gradient(LEFT.value),
                                  lossType=lossTypes.Absolute => Abs_Gradient(LEFT.value),
                                  lossType=lossTypes.Huber => Huber_Gradient(LEFT.value),
                                  0);
                            SELF := LEFT), LOCAL);
    RETURN gradients;
  END;
  // Function to calculate the residuals given the current predictions and the
  // actual target values.
  SHARED CalcResiduals(DATASET(NumericField) pred, DATASET(BfTreeNodeDat) actuals) := FUNCTION
    resids := JOIN(pred, actuals, LEFT.wi = RIGHT.wi AND LEFT.id = RIGHT.id,
                          TRANSFORM(RECORDOF(RIGHT),
                                    SELF.value := RIGHT.value - LEFT.value,
                                    SELF := RIGHT), LOCAL);
    RETURN resids;
  END;
  // Function to calculate the variance of the dependent data.
  SHARED DATASET(NumericField) CalcVar(DATASET(BfTreeNodeDat) ds) := FUNCTION
    vars0 := TABLE(ds, {wi, number, REAL var := VARIANCE(GROUP, value)}, wi, number);
    vars := PROJECT(vars0, TRANSFORM(NumericField,
                      SELF.value := LEFT.var,
                      SELF.wi := LEFT.wi,
                      SELF.number := LEFT.number,
                      SELF.id := 0), LOCAL);
    RETURN vars;
  END;
  // Return the BfTreeNodeDat records representing the learned Boosted Forest.
  EXPORT DATASET(BfTreeNodeDat) GetNodes := FUNCTION
    X := X_in;
    // Distribute the data by work-item and record-id.
    Y := DISTRIBUTE(Y_in, HASH32(wi, id));
    // Transform to the BfTreeNodeDat format that is needed for the loop.
    initData := PROJECT(Y, TRANSFORM(BfTreeNodeDat,
                                  SELF.treeId := 0,
                                  SELF.nodeId := 0,
                                  SELF.parentId := 0,
                                  SELF.isLeft := FALSE,
                                  SELF.level := 0,
                                  SELF.origId := 0,
                                  SELF.ir := 0,
                                  SELF.bfLevel := 0,
                                  SELF.depend := 0,
                                  SELF := LEFT), LOCAL);
    // Calculate the original variance.
    Orig_Var := CalcVar(initData);
    // Input to the loop is the set of tree nodes learned so far plus the Y values
    // to be matched.  The two can be separated on (id = 0); Zero for tree nodes.
    // The first time through the loop, there are no tree nodes, and the Y data
    // to be matched is the original Y data.
    // On each successive loop, the tree nodes from the tree just learned are added
    // to the loop data and the Y data to be matched are the gradient on the residual.
    doOneLevel(DATASET(BfTreeNodeDat) loopDat, UNSIGNED loopLevel) := FUNCTION
      // Extract the Y values from the loopDat.  These are the original Y values on the first
      // iteration, and the residuals on subsequent iterations.
      Y_nodes := loopDat;
      Y_Var := CalcVar(Y_nodes);  // Variance of Residuals for each wi.
      // Calculate the Residual Variance Ratio (RVR), and eliminate any WIs that
      // have RVR below the earlyStopThreshold.
      RVR := JOIN(Y_Var, Orig_var, LEFT.wi = RIGHT.wi,
                      TRANSFORM(RECORDOF(LEFT),
                          SELF.value :=  LEFT.value / RIGHT.value,
                          SELF.id := IF(SELF.value <= earlyStopThreshold, SKIP, 0),
                          SELF := LEFT), LOOKUP);
      // Filter out any WI's that have met the earlyStopThreshold from Y and X data
      // and convert to GenField format for Random Forest.
      Y_filt := JOIN(Y_nodes, RVR, LEFT.wi = RIGHT.wi, TRANSFORM(GenField, SELF := LEFT), LOOKUP);
      X_filt := JOIN(X, RVR, LEFT.wi = RIGHT.wi, TRANSFORM(LEFT), LOOKUP);
      // Calculate the gradient for training the forest for this level
      Y_grad0 := CalcGradients(Y_filt);
      Y_grad := IF(loopLevel = 1, Y_filt, Y_grad0); // Do use gradients on the first pass.
      // Create a random forest for any remaining work-items
      // featuresPerNode = 0 says use the default features per level
      forest := RF_Regression(X_filt, Y_grad, numTrees:=forestSizeAdj, featuresPerNode:=fpn,
                                            maxDepth:=maxTreeDepth);
      // Get the nodes for the forest just constructed
      newNodes := forest.GetNodes;
      newNodesExt := PROJECT(newNodes,
                         TRANSFORM(BfTreeNodeDat,
                           SELF.bfLevel := loopLevel,
                           SELF := LEFT), LOCAL);
      // Use the nodes to predict values for each X in the training set
      predictions := forest.ForestPredict(newNodes, X_filt);
      residuals := CalcResiduals(predictions, Y_nodes);
      // If we've reached the early stopping criterion for all work units, then
      // return an empty dataset to stop the loop.
      outDataContinue := newNodesExt + residuals;
      outDataFinished := DATASET([], BfTreeNodeDat);
      BOOLEAN finished := NOT EXISTS(Y_filt);
      outData := IF(finished, outDataFinished, outDataContinue);
      RETURN outData;
    END;
    //finalOut := LOOP(initData, maxLevels, doOneLevel(ROWS(LEFT), COUNTER));
    finalOut := LOOP(initData, LEFT.id <> 0, EXISTS(ROWS(LEFT)(COUNTER <= maxLevels)),
                      doOneLevel(ROWS(LEFT), COUNTER));
    //finalOut := doOneLevel(initData, 1);
    finalNodes := finalOut(id = 0);
    //finalNodes := finalOut;
    finalResiduals := finalOut(id > 0);
    RETURN finalNodes;
  END;
  EXPORT DATASET(NumericField) BfPredict(DATASET(BfTreeNodeDat) nodes, DATASET(NumericField) X) := FUNCTION
    // Adjust the work items so that we can compute all the forests in parallel, and
    // convert them to the Forest Model's TreeNodeDat format for consumption by the LearningForest
    maxLevs := TABLE(nodes, {wi, UNSIGNED maxLevel := MAX(GROUP, bfLevel)}, wi);
    newWIs := NORMALIZE(maxLevs, LEFT.maxLevel, TRANSFORM({UNSIGNED wi, UNSIGNED bfLevel, UNSIGNED maxLevel, UNSIGNED newWI},
                                        SELF.newWI := ((LEFT.wi - 1) * LEFT.maxLevel) + COUNTER,
                                        SELF.bfLevel := COUNTER,
                                        SELF := LEFT));
    forestNodes := JOIN(nodes, newWIs, LEFT.wi = RIGHT.wi AND LEFT.bfLevel = RIGHT.bfLevel,
                          TRANSFORM(TreeNodeDat,
                                SELF.wi := RIGHT.newWI,
                                SELF := LEFT), LOOKUP, FEW);
    // Now replicate the X values to each new work item, so that we can run all trees
    // in parallel.
    X_repl := JOIN(X, newWIs, LEFT.wi = RIGHT.wi, TRANSFORM(GenField,
                                              SELF.wi := RIGHT.newWI,
                                              SELF.isOrdinal := FALSE,
                                              SELF := LEFT), MANY, LOOKUP);
    // Run all of the forest regressions
    myRF := RF_Regression();
    pred := myRF.ForestPredict(forestNodes, X_repl);
    // Now add up all of the regression results for each original work-item.
    // First, convert the wi's back to their original values.
    pred_adj0 := JOIN(pred, newWIs, LEFT.wi = RIGHT.newWI, TRANSFORM(RECORDOF(LEFT),
                                SELF.wi := RIGHT.wi,
                                SELF := LEFT), LOOKUP, FEW);
    // Rollup the results by work item for each id
    pred_adj := SORT(DISTRIBUTE(pred_adj0, HASH32(wi, id)), wi, id, LOCAL);
    pred_out := ROLLUP(pred_adj, TRANSFORM(NumericField,
                                    SELF.value := LEFT.value + RIGHT.value,
                                    SELF := LEFT), wi, id, LOCAL);
    RETURN pred_out;
  END;
  /**
    * Predict a series of regression values based on the provided model.
    *
    */
  EXPORT DATASET(NumericField) Predict(DATASET(Layout_Model2) mod, DATASET(NumericField) X) := FUNCTION
    nodes := Model2Nodes(mod);
    pred :=  BfPredict(nodes, X);
    RETURN pred;
  END;
  /**
    * Train and Retrieve the model
    *
    */
  EXPORT GetModel := FUNCTION
    nodes := GetNodes;
    mod := Nodes2Model(nodes);
    RETURN mod;
  END;
  /**
    * Get summary statistical information about the model.
    *
    * @param mod A model previously returned from GetModel.
    * @return A single ModelStats record per work-item, containing information about the model
    *         for that work-item.
    * @see LT_Types.ModelStats
    */
  EXPORT DATASET(ModelStats) GetModelStats(DATASET(Layout_Model2) mod) := FUNCTION
    myRF := RF_Base();
    doOneLevel(DATASET(ModelStats) recs, UNSIGNED c) := FUNCTION
      thisMod := mod(indexes[2] = c);
      // Strip off the first two indexes to format as RF model.
      mod_adj := PROJECT(thisMod, TRANSFORM(RECORDOF(LEFT),
                              SELF.indexes := LEFT.indexes[3 ..],
                              SELF := LEFT), LOCAL);
      rfModStats := myRF.GetModelStats(mod_adj);
      outStats := PROJECT(rfModStats, TRANSFORM(RECORDOF(LEFT),
                                          SELF.bfLevel := c,
                                          SELF := LEFT), LOCAL);
      RETURN recs + outStats;
    END;
    maxLevel := MAX(mod, indexes[2]);
    initRecs := DATASET([], ModelStats);
    rfRecs := LOOP(initRecs, maxLevel, doOneLevel(ROWS(LEFT), COUNTER));
    rfRecs_S := SORT(rfRecs, wi, bfLevel);
    summary := TABLE(rfRecs_S, {wi, xtreeCount := SUM(GROUP, treeCount),
                                    xminTreeDepth := MIN(GROUP, minTreeDepth),
                                    xmaxTreeDepth := MAX(GROUP, maxTreeDepth),
                                    xavgTreeDepth := AVE(GROUP, avgTreeDepth),
                                    xminTreeNodes := MIN(GROUP, minTreeNodes),
                                    xmaxTreeNodes := MAX(GROUP, maxTreeNodes),
                                    xavgTreeNodes := AVE(GROUP, avgTreeNodes),
                                    xtotalNodes := SUM(GROUP, totalNodes),
                                    xminSupport := MIN(GROUP, minSupport),
                                    xmaxSupport := MAX(GROUP, maxSupport),
                                    xavgSupport := AVE(GROUP, avgSupport),
                                    xmaxSupportPerLeaf := MAX(GROUP, maxSupportPerLeaf),
                                    xavgSupportPerLeaf := AVE(GROUP, avgSupportPerLeaf),
                                    xavgLeafDepth := AVE(GROUP, avgLeafDepth),
                                    xminLeafDepth := MIN(GROUP, minLeafDepth),
                                    xbfLevel := MAX(GROUP, bfLevel)}, wi);
    outRecs := PROJECT(summary, TRANSFORM(ModelStats,
                                    SELF.wi := LEFT.wi,
                                    SELF.treeCount := LEFT.xtreeCount,
                                    SELF.minTreeDepth := LEFT.xminTreeDepth,
                                    SELF.maxTreeDepth := LEFT.xmaxTreeDepth,
                                    SELF.avgTreeDepth := LEFT.xavgTreeDepth,
                                    SELF.minTreeNodes := LEFT.xminTreeNodes,
                                    SELF.maxTreeNodes := LEFT.xmaxTreeNodes,
                                    SELF.avgTreeNodes := LEFT.xavgTreeNodes,
                                    SELF.totalNodes := LEFT.xtotalNodes,
                                    SELF.minSupport := LEFT.xminSupport,
                                    SELF.maxSupport := LEFT.xmaxSupport,
                                    SELF.avgSupport := LEFT.xavgSupport,
                                    SELF.maxSupportPerLeaf := LEFT.xmaxSupportPerLeaf,
                                    SELF.avgSupportPerLeaf := LEFT.xavgSupportPerLeaf,
                                    SELF.avgLeafDepth := LEFT.xavgLeafDepth,
                                    SELF.minLeafDepth := LEFT.xminLeafDepth,
                                    SELF.bfLevel := LEFT.xbfLevel),LOCAL);
    RETURN outRecs;
  END;
END;