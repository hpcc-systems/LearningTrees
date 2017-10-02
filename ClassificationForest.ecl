IMPORT $ AS LT;
IMPORT LT.LT_Types AS Types;
IMPORT ML_Core;
IMPORT ML_Core.Types as CTypes;
IMPORT LT.internal AS int;


NumericField := CTypes.NumericField;
DiscreteField := CTypes.DiscreteField;
Layout_Model2 := Types.Layout_Model2;
TreeNodeDat := Types.TreeNodeDat;
t_Discrete := CTypes.t_Discrete;
t_Work_Item := CTypes.t_Work_Item;
t_RecordID := CTypes.t_RecordId;
ClassProbs := Types.ClassProbs;

/**
  * Classification Forest
  *
  * Classification using Random Forest.
  * This module implements Random Forest classification as described by
  * Breiman, 2001 with extensions.
  * (see https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
  *
  * Random Forests provide a very effective method for classification
  * with few assumptions about the nature of the data.  They are known
  * to be one of the best out-of-the-box methods as there are few assumptions
  * made regarding the nature of the data or its relationship to classes.
  * Random Forests can effectively manage large numbers
  * of features, and will automatically choose the most relevant features.
  * Random Forests inherently support multi-class problems.  Any number of
  * class labels can be used.
  *
  * This implementation supports both Numeric (discrete or continuous) and
  * Nominal (unordered categorical values) for the independent (X) features.
  * There is therefore, no need to one-hot encode categorical features.
  * Nominal features should be identified by including their feature 'number'
  * in the set of 'nominalFields' in GetModel.
  *
  * @param numTrees The number of trees to create as the forest for each work-item.
  *                 This defaults to 100, which is adequate for most cases.
  * @param featuresPerNode The number of features to choose among at each split in
  *                 each tree.  This number of features will be chosen at random
  *                 from the full set of features.  The default is the square
  *                 root of the number of features provided, which works well
  *                 for most cases.
  * @param maxDepth The deepest to grow any tree in the forest.  The default is
  *                 64, which is adequate for most purposes.  Increasing this value
  *                 for very large and complex problems my provide slightly greater
  *                 accuracy at the expense of much greater runtime.
  */
  EXPORT ClassificationForest(UNSIGNED numTrees=100,
              UNSIGNED featuresPerNode=0,
              UNSIGNED maxDepth=64) := MODULE(LT.LearningForest(numTrees, featuresPerNode, maxDepth))
    /**
      * Fit a model that maps independent data (X) to its class (Y).
      *
      * @param X  The set of independent data in NumericField format
      * @param Y  The set of classes in DiscreteField format that correspond to the independent data
      *           i.e. same 'id'.
      * @param nominalFields An optional set of field 'numbers' that represent Nominal (i.e. unordered,
      *                      categorical) values.  Specifying the nominal fields improves run-time
      *                      performance on these fields and my improve accuracy as well.  Binary fields
      *                      (fields with only two values) need not be listed here as they can be
      *                      considered either ordinal or nominal.
      * @return Model in Layout_Model2 format describing the fitted forest.
      */
    EXPORT DATASET(Layout_Model2) GetModel(DATASET(NumericField) X, DATASET(DiscreteField) Y, SET OF UNSIGNED nominalFields=[]) := FUNCTION
      genX := NF2GenField(X, nominalFields);
      genY := DF2GenField(Y);
      myRF := int.RF_Classification(genX, genY, numTrees, featuresPerNode, maxDepth);
      model := myRF.GetModel;
      RETURN model;
    END;
    /**
      * Classify a set of data points using a previously fitted model
      *
      * @param X The set of independent data to classify in NumericField format
      * @param mod A model previously returned by GetModel in Layout_Model2 format
      * @return A DiscreteField dataset that indicates the class of each item in X
      */
    EXPORT DATASET(DiscreteField) Classify(DATASET(NumericField) X, DATASET(Layout_Model2) mod,
                                              BOOLEAN balanceClasses=FALSE) := FUNCTION
      genX := NF2GenField(X);
      myRF := int.RF_Classification();
      classes := myRF.Classify(genX, mod, balanceClasses);
      RETURN classes;
    END;

    /**
      * Get class probabilities
      *
      * Calculate the 'probability' that each datapoint is in each class.
      * Probability is used loosely here, as the proportion of trees that
      * voted for each class for each datapoint.
      * @param X The set of independent data to classify in NumericField format
      * @param mod A model previously returned by GetModel in Layout_Model2 format
      * @return DATASET(ClassProbs), one record per datapoint (i.e. id) per class
      *         label.  Class labels with zero votes are omitted.
      */
    EXPORT DATASET(ClassProbs) GetClassProbs(DATASET(NumericField) X, DATASET(Layout_Model2) mod,
                                              BOOLEAN balanceClasses=FALSE) := FUNCTION
      genX := NF2GenField(X);
      myRF := int.RF_Classification();
      probs := myRF.GetClassProbs(genX, mod, balanceClasses);
      probsS := SORT(probs, wi, id, class); // Global sort
      RETURN probsS;
    END;

    /**
      * Extract the set of tree nodes from a model
      *
      * @param mod A model as returned from GetModel
      * @return Set of tree nodes representing the fitted forest in DATASET(TreeNodeDat) format
      */
    EXPORT DATASET(TreeNodeDat) Model2Nodes(DATASET(Layout_Model2) mod) := FUNCTION
      myRF := int.RF_Classification();
      nodes0 := myRF.Model2Nodes(mod);
      nodes := SORT(nodes0, wi, treeId, level, nodeId, LOCAL);
      RETURN nodes;
    END;
    /**
      * Extract the set of class weights from the model
      *
      * @param mod A model as returned from GetModel
      * @return DATASET(classWeightRec) representing weight for each class label
      */
    EXPORT  Model2ClassWeights(DATASET(Layout_Model2) mod) := FUNCTION
      myRF := int.RF_Classification();
      cw := myRF.Model2ClassWeights(mod);
      RETURN cw;
    END;
    /**
      * Get statistics about the accuracy of the classification
      *
      * Provides accuracy statistics as follows:
      * - errCount -- The number of misclassified samples
      * - errPct -- The percentage of samples that were misclasified (0.0 - 1.0)
      * - RawAccuracy -- The percentage of samples properly classified (0.0 - 1.0)
      * - PoD -- Power of Discrimination.  Indicates how this classification performed
      *           relative to a random guess of class.  Zero or negative indicates that
      *           the classification was no better than a random guess.  1.0 indicates a
      *           perfect classification.  For example if there are two equiprobable classes,
      *           then a random guess would be right about 50% of the time.  If this
      *           classification had a Raw Accuracy of 75%, then its PoD would be .5
      *           (half way between a random guess and perfection).
      * - PoDE -- Power of Discrimination Extended.  Indicates how this classification
      *           performed relative to guessing the most frequent class (i.e. the trivial
      *           solution).  Zero or negative indicates that this classification is no
      *           better than the trivial solution.  1.0 indicates perfect classification.
      *           For example, if 95% of the samples were of class 1, then the trivial
      *           solution would be right 95% of the time.  If this classification had a
      *           raw accuracy of 97.5%, its PoDE would be .5 (i.e. half way between
      *           trivial solution and perfection).
      * Normally, this should be called using data samples that were not included in the
      * training set.  In that case, these statistics are considered Out-of-Sample error
      * statistics.  If it is called with the X and Y from the training set, it provides
      * In-Sample error stats, which should never be used to rate the classification model.
      *
      * @param X The independent data in DATASET(NumericField) format
      * @param Y The corresponding class labels in DATASET(DiscreteField) format
      * @return Dataset containing one record per work-item containing the metrics described above
      */
    EXPORT GetErrorStats(DATASET(NumericField) X, DATASET(DiscreteField) Y, DATASET(Layout_Model2) mod,
                                        BOOLEAN balanceClasses=FALSE) := FUNCTION
      myRF := int.RF_Classification();
      predClasses := SORT(DISTRIBUTE(Classify(X, mod, balanceClasses), HASH32(wi, id)), wi, id, LOCAL);
      actualClasses := SORT(DISTRIBUTE(Y, HASH32(wi, id)), wi, id, LOCAL);
      cmp := JOIN(predClasses, actualClasses, LEFT.wi = RIGHT.wi AND LEFT.id = RIGHT.id,
                    TRANSFORM({t_Work_Item wi, t_RecordID id, t_Discrete pred, t_Discrete actual, UNSIGNED errs},
                                SELF.pred := LEFT.value, SELF.actual := RIGHT.value,
                                SELF.errs := IF(SELF.pred = SELF.actual, 0, 1), SELF := LEFT), LOCAL);
      errCnts := TABLE(cmp, {wi, UNSIGNED errCnt := SUM(GROUP, errs)}, wi);
      classCounts := TABLE(actualClasses, {wi, value, UNSIGNED valCount := COUNT(GROUP)}, wi, value);
      wiClassInfo := TABLE(classCounts, {wi, UNSIGNED numClasses := COUNT(GROUP), UNSIGNED recCount := SUM(GROUP, valCount),
                          UNSIGNED maxCount := MAX(GROUP, valCount)}, wi);
      errStats := JOIN(errCnts, wiClassInfo, LEFT.wi = RIGHT.wi,
                        TRANSFORM({t_Work_Item wi, UNSIGNED errCount, REAL errPct, REAL rawAccuracy, REAL PoD, REAL PodE},
                                  SELF.wi := LEFT.wi, SELF.errCount := LEFT.errCnt,
                                  SELF.errPct := SELF.errCount / RIGHT.recCount;
                                  SELF.rawAccuracy := 1 - SELF.errPct,
                                  SELF.PoD := (SELF.rawAccuracy - 1/RIGHT.numClasses) / (1-1/RIGHT.numClasses),
                                  SELF.PoDE := (SELF.rawAccuracy - RIGHT.maxCount / RIGHT.recCount) / (1 - RIGHT.maxCount / RIGHT.recCount)));
      RETURN errStats;
    END;
    /**
      * Get the confusion matrix for the given model and test data.
      *
      * The confusion matrix indicates the number of datapoints that were classified correctly or incorrectly
      * for each class label.
      * The matrix is provided as a matrix of size numClasses x numClasses as with fields asfollows:
      * - 'wi' -- The work item id
      * - 'pred' -- the predicted class label (from Classify)
      * - 'actual' -- the actual (target) class label
      * - 'samples' -- the count of samples that were predicted as 'pred', but should have been 'actual'
      * - 'totSamples' -- the total number of samples that were predicted as 'pred'
      * - 'pctSamples' -- the percentage of all samples that were predicted as 'pred', that should
      *                have been 'actual' (i.e. samples / totSamples)
      *
      * This is a useful tool for understanding how the algorithm achieved the overall accuracy.  For example:
      * were the common classes mostly correct, while less common classes often misclassified?  Which
      * classes were most often confused?
      *
      * This should be called with test data that is independent of the training data in order to understand
      * the out-of-sample (i.e. generalization) performance.
      *
      * @param X The independent data in DATASET(NumericField) format
      * @param Y The expected classes of the X samples in DATASET(DiscreteField) format
      * @param mod The model as returned from GetModel
      * @returns The confusion matrix as described above
      *
      */
    EXPORT ConfusionMatrix(DATASET(NumericField) X, DATASET(DiscreteField) Y, DATASET(Layout_Model2) mod,
                                    BOOLEAN balanceClasses=FALSE) := FUNCTION
      myRF := int.RF_Classification();
      predClasses := SORT(DISTRIBUTE(Classify(X, mod, balanceClasses), HASH32(wi, id)), wi, id, LOCAL);
      actualClasses := SORT(DISTRIBUTE(Y, HASH32(wi, id)), wi, id, LOCAL);
      // Create a record for each combination of predicted and actual as a DiscreteField matrix
      predxactual := JOIN(predClasses, actualClasses, LEFT.wi = RIGHT.wi AND LEFT.id = RIGHT.id,
                    TRANSFORM(DiscreteField,
                                SELF.id := LEFT.value, SELF.number := RIGHT.value,
                                SELF.value := 0,
                                SELF := LEFT), LOCAL);
      // Summarize to a record per class combination encountered with the number of occurrences of that combination
      confusion := TABLE(predxactual, {wi, id, number, UNSIGNED value := COUNT(GROUP)}, wi, id, number);
      // Calculate the total number of records for each predicted value
      predSummary := TABLE(confusion, {wi, id, UNSIGNED samples := SUM(GROUP, value)}, wi, id);

      confSummary := JOIN(confusion, predSummary, LEFT.wi = RIGHT.wi AND LEFT.id = RIGHT.id,
                              TRANSFORM({UNSIGNED wi,
                                UNSIGNED pred,
                                UNSIGNED actual,
                                UNSIGNED samples,
                                UNSIGNED totSamples,
                                REAL pctSamples},
                                SELF.pred := LEFT.id,
                                SELF.actual := LEFT.number,
                                SELF.samples := LEFT.value, SELF.pctSamples := SELF.samples / RIGHT.samples,
                                SELF.totSamples := RIGHT.samples, SELF := LEFT), LOOKUP);
      RETURN SORT(confSummary, wi, pred, actual);
    END;
  END;