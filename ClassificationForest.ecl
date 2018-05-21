/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2017 HPCC Systems®.  All rights reserved.
############################################################################## */
IMPORT $ AS LT;
IMPORT LT.LT_Types AS Types;
IMPORT ML_Core;
IMPORT ML_Core.Types as CTypes;
IMPORT ML_Core.interfaces AS Interfaces;
IMPORT LT.internal AS int;

NumericField := CTypes.NumericField;
DiscreteField := CTypes.DiscreteField;
Layout_Model2 := CTypes.Layout_Model2;
TreeNodeDat := Types.TreeNodeDat;
t_Discrete := CTypes.t_Discrete;
t_Work_Item := CTypes.t_Work_Item;
t_RecordID := CTypes.t_RecordId;
ClassProbs := Types.ClassProbs;
IClassify2 := Interfaces.IClassify2;


/**
  * Classification using Random Forest algorithm.
  * <p>This module implements Random Forest classification as described by
  * Breiman, 2001 with extensions.
  * (see https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
  *
  * <p>Random Forests provide a very effective method for classification
  * with few assumptions about the nature of the data.  They are known
  * to be one of the best out-of-the-box methods as there are few assumptions
  * made regarding the nature of the data or its relationship to classes.
  * Random Forests can effectively manage large numbers
  * of features, and will automatically choose the most relevant features.
  * Random Forests inherently support multi-class problems.  Any number of
  * class labels can be used.
  *
  * <p>This implementation supports both Numeric (discrete or continuous) and
  * Nominal (unordered categorical values) for the independent (X) features.
  * There is therefore, no need to one-hot encode categorical features.
  * Nominal features should be identified by including their feature 'number'
  * in the set of 'nominalFields' in GetModel.
  *
  * <p>RegressionForest supports the Myriad interface meaning that multiple
  * independent models can be computed with a single call (see ML_Core.Types
  * for information on using the Myriad feature).
  *
  * <p>Notes on use of NumericField and DiscreteField layouts:
  * <ul>
  * <li>Work-item ids ('wi' field) are not required to be sequential, though they must be positive
  *   numbers.  It is a good practice to assign wi = 1 when only one work-item is used.</li>
  * <li>Record Ids ('id' field) are not required to be sequential, though slightly faster performance
  *   will result if they are sequential (i.e. 1 .. numRecords) for each work-item.</li>
  * <li>Feature numbers ('number' field) are not required to be sequential, though slightly faster
  *   performance will result if they are (i.e. 1 .. numFeatures) for each work-item.</li>
  * </ul>
  *
  * @param numTrees The number of trees to create in the forest for each work-item.
  *                 This defaults to 100, which is adequate for most cases.  Increasing
  *                 this parameter generally results in less variance in accuracy between
  *                 runs, at the expense of greater run time.
  * @param featuresPerNode The number of features to choose among at each split in
  *                 each tree.  This number of features will be chosen at random
  *                 from the full set of features.  The default (0) uses the square
  *                 root of the number of features provided, which works well
  *                 for most cases.
  * @param maxDepth The deepest to grow any tree in the forest.  The default is
  *                 100, which is adequate for most purposes.  Increasing this value
  *                 for very large and complex problems my provide slightly greater
  *                 accuracy at the expense of much greater runtime.
  * @param nominalFields An optional set of field 'numbers' that represent Nominal (i.e. unordered,
  *                      categorical) values.  Specifying the nominal fields improves run-time
  *                      performance on these fields and may improve accuracy as well.  Binary fields
  *                      (fields with only two values) need not be included here as they can be
  *                      considered either ordinal or nominal.  The default is to treat all fields as
  *                      ordered.
  *                      Note that this feature should only be used if all of the independent
  *                      data for all work-items use the same record format, and therefore have the same
  *                      set of nominal fields.
  * @param balanceClasses An optional Boolean parameter.  If true, it indicates that the voting among
  *                       trees should be biased inversely to the frequency of the class for which it
  *                       is voting.  This may help in scenarios where there are far more samples of
  *                       certain classes than of others.  The default is to not balance (i.e. FALSE).
  */
  EXPORT ClassificationForest(UNSIGNED numTrees=100,
              UNSIGNED featuresPerNode=0,
              UNSIGNED maxDepth=100,
              SET OF UNSIGNED nominalFields=[],
              BOOLEAN balanceClasses=FALSE) := MODULE(LT.LearningForest(numTrees, featuresPerNode, maxDepth), IClassify2)
    /**
      * Fit and return a model that maps independent data (X) to its predicted class (Y).
      *
      * @param independents  The set of independent data in NumericField format.
      * @param dependents The set of classes in DiscreteField format that correspond to the independent data
      *           i.e. same 'id'.
      * @param nominalFields An optional set of field 'numbers' that represent Nominal (i.e. unordered,
      *                      categorical) values.  Specifying the nominal fields improves run-time
      *                      performance on these fields and my improve accuracy as well.  Binary fields
      *                      (fields with only two values) need not be listed here as they can be
      *                      considered either ordinal or nominal.  Example: [3,5,7].
      * @return Model in Layout_Model2 format describing the fitted forest.
      * @see ML_Core.Types.NumericField, ML_Core.Types.DiscreteField, ML_Core.Types.Layout_Model2
      */
    EXPORT DATASET(Layout_Model2) GetModel(DATASET(NumericField) independents, DATASET(DiscreteField) dependents) := FUNCTION
      genX := NF2GenField(independents, nominalFields);
      genY := DF2GenField(dependents);
      myRF := int.RF_Classification(genX, genY, numTrees, featuresPerNode, maxDepth);
      model := myRF.GetModel;
      RETURN model;
    END;
    /**
      * Classify a set of data points using a previously fitted model
      *
      * @param model A model previously returned by GetModel in Layout_Model2 format.
      * @param observations The set of independent data to classify in NumericField format.
      * @return A DiscreteField dataset that indicates the predicted class of each item
      * in observations.
      */
    EXPORT DATASET(DiscreteField) Classify(DATASET(Layout_Model2) model, DATASET(NumericField) observations) := FUNCTION
      genX := NF2GenField(observations);
      myRF := int.RF_Classification();
      classes := myRF.Classify(genX, model, balanceClasses);
      RETURN classes;
    END;

    /**
      * Calculate the 'probability' that each data point is in each class.
      * <p>Probability is approximated by computing the proportion of trees that
      * voted for each class for each data point, so should not be treated
      * as a reliable measure of true probability.
      *
      * @param model A model previously returned by GetModel in Layout_Model2 format.
      * @param observations The set of independent data to classify in NumericField format.
      * @return DATASET(ClassProbs), one record per datapoint (i.e. id) per class
      *         label.  Class labels with zero votes are omitted.
      * @see LT_Types.ClassProbs
      *
      */
    EXPORT DATASET(ClassProbs) GetClassProbs(DATASET(Layout_Model2) model, DATASET(NumericField) observations) := FUNCTION
      genX := NF2GenField(observations);
      myRF := int.RF_Classification();
      probs := myRF.GetClassProbs(genX, model, balanceClasses);
      probsS := SORT(probs, wi, id, class); // Global sort
      RETURN probsS;
    END;

    /**
      * Extract the set of class weights from the model.
      * <p>Classes are weighted inversely proportional to their frequency in the
      * training data.  <p>Note that the class weights are based on a non-linear
      * 'proportion' to avoid excess weight for classes with very low frequency.
      * <p>These weights are only used when the 'balanceClasses' option is TRUE.
      *
      * @param mod A model as returned from GetModel.
      * @return DATASET(ClassWeightRec) representing weight for each class label.
      * @see LT_Types.ClassWeightRec
      */
    EXPORT  Model2ClassWeights(DATASET(Layout_Model2) mod) := FUNCTION
      myRF := int.RF_Classification();
      cw := myRF.Model2ClassWeights(mod);
      RETURN cw;
    END;
  END;
