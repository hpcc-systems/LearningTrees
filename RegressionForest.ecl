/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2017 HPCC Systems®.  All rights reserved.
############################################################################## */

IMPORT ML_Core;
IMPORT ML_Core.Types as CTypes;
IMPORT ML_Core.interfaces AS Interfaces;
IMPORT $ AS LT;
IMPORT LT.LT_Types AS Types;
IMPORT LT.internal AS int;


NumericField := CTypes.NumericField;
DiscreteField := CTypes.DiscreteField;
Layout_Model2 := CTypes.Layout_Model2;
IRegression2 := Interfaces.IRegression2;

/**
  * Regression using Random Forest algorithm.
  * This module implements Random Forest regression as described by
  * Breiman, 2001 with extensions
  * (see https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf).
  *
  * <p>Random Forests provide an effective method for regression.  They are known
  * to be one of the best out-of-the-box methods as there are few assumptions
  * made regarding the nature of the data or its relationships.
  * Random Forests can effectively manage large numbers
  * of features, and will automatically choose the most relevant features.
  *
  * <p>Regression Forests can handle non-linear and discontinuous relationships
  * among features.
  *
  * <p>One limitation of Regression Forests is that they provide no extrapolation
  * beyond the bounds of the training data.  The training set should extend to the
  * limits of expected feature values.
  *
  * <p>This implementation allows both Ordinal (discrete or continuous) and
  * Nominal (unordered categorical values) for the independent (X) features.
  * There is therefore, no need to one-hot encode categorical features.
  * Nominal features should be identified by including their feature 'number'
  * in the set of 'nominalFields'.
  *
  * <p>RegressionForest supports the Myriad interface meaning that multiple
  * independent models can be computed with a single call (see ML_Core.Types
  * for information on using the Myriad feature).
  *
  * <p>Notes on use of NumericField layouts:
  * <ul>
  * <li>Work-item ids ('wi' field) are not required to be sequential, though they must be positive
  *   numbers.  It is a good practice to assign wi = 1 when only one work-item is used.</li>
  * <li>Record Ids ('id' field) are not required to be sequential, though slightly faster performance
  *   will result if they are sequential (i.e. 1 .. numRecords) for each work-item.</li>
  * <li>Feature numbers ('number' field) are not required to be sequential, though slightly faster
  *   performance will result if they are (i.e. 1 .. numFeatures) for each work-item.</li>
  * </ul>
  *
  * @param numTrees The number of trees to create as the forest for each work-item.
  *                 This defaults to 100, which is adequate for most cases.  Increasing
  *                 this parameter generally results in less variance in accuracy between
  *                 runs, at the expense of greater run time.
  * @param featuresPerNode The number of features to choose among at each split in
  *                 each tree.  This number of features will be chosen at random
  *                 from the full set of features.  The default value (0) uses the square
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
  *                      ordered.  Note that this feature should only be used if all of the independent
  *                      data for all work-items use the same record format, and therefore have the same
  *                      set of nominal fields.
  */
  EXPORT RegressionForest(UNSIGNED numTrees=100,
              UNSIGNED featuresPerNode=0,
              UNSIGNED maxDepth=100,
              SET OF UNSIGNED nominalFields=[]) := MODULE(LT.LearningForest(numTrees, featuresPerNode, maxDepth), IRegression2)
    /**
      * Fit a model that maps independent data (X) to its class (Y).
      *
      * @param independents  The set of independent data in NumericField format.
      * @param dependents  The dependent variable in NumericField format.  The 'number' field is not used as
      *           only one dependent variable is currently supported. For consistency, it should be set to 1.
      * @return Model in Layout_Model2 format describing the fitted forest.
      * @see ML_Core.Types.NumericField, ML_Core.Types.Layout_Model2
      */
    EXPORT  GetModel(DATASET(NumericField) independents, DATASET(NumericField) dependents) := FUNCTION
      genX := NF2GenField(independents, nominalFields);
      genY := NF2GenField(dependents);
      myRF := int.RF_Regression(genX, genY, numTrees, featuresPerNode, maxDepth);
      model := myRF.GetModel;
      RETURN model;
    END;
    /**
      * Classify a set of data points using a previously fitted model.
      *
      * @param mod A model previously returned by GetModel in Layout_Model2 format.
      * @param observations The set of independent data in NumericField format.
      * @return A NumericField dataset that provides a prediction for each record in observations.
      */
    EXPORT DATASET(NumericField) Predict(DATASET(Layout_Model2) model, DATASET(NumericField) observations) := FUNCTION
      genX := NF2GenField(observations);
      myRF := int.RF_Regression();
      predictions := myRF.Predict(genX, model);
      RETURN predictions;
    END;
  END;
