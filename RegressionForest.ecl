
IMPORT ML_Core;
IMPORT ML_Core.Types as CTypes;
IMPORT LT_Types AS Types;
IMPORT $ AS LT;
IMPORT LT.internal AS int;


NumericField := CTypes.NumericField;
DiscreteField := CTypes.DiscreteField;
Layout_Model2 := Types.Layout_Model2;

/**
  * Regression Forest
  *
  * Regression using Random Forest.
  * This module implements Random Forest regression as described by
  * Breiman, 2001 with extensions.
  * (see https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
  *
  * Random Forests provide an effective method for regression
  * with few assumptions about the nature of the data.  They are known
  * to be one of the best out-of-the-box methods as there are few assumptions
  * made regarding the nature of the data or its relationships.
  * Random Forests can effectively manage large numbers
  * of features, and will automatically choose the most relevant features.
  *
  * Regression Forests can handle non-linear and discontinuous relationships
  * among features.
  *
  * One limitation of Regression Forests is that they provide no extrapolation
  * beyond the bounds of the training data.  The training set should extend to the
  * limits of expected feature values.
  *
  * This implementation allows both Ordinal (discrete or continuous) and
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
  EXPORT RegressionForest(UNSIGNED numTrees=100,
              UNSIGNED featuresPerNode=0,
              UNSIGNED maxDepth=64) := MODULE(LT.LearningForest(numTrees, featuresPerNode, maxDepth))
    /**
      * Fit a model that maps independent data (X) to its class (Y).
      *
      * @param X  The set of independent data in NumericField format
      * @param Y  The dependent variable in NumericField format.  The 'number' field is not used as
      *           only one dependent variable is currently supported. For consistency, it should be set to 1.
      * @param nominalFields An optional set of field 'numbers' that represent Nominal (i.e. unordered,
      *                      categorical) values.  Specifying the nominal fields improves run-time
      *                      performance on these fields and my improve accuracy as well.  Binary fields
      *                      (fields with only two values) need not be included here as they can be
      *                      considered either ordinal or nominal.
      * @return Model in Layout_Model2 format describing the fitted forest.
      */
    EXPORT  GetModel(DATASET(NumericField) X, DATASET(NumericField) Y, SET OF UNSIGNED nominalFields=[]) := FUNCTION
      genX := NF2GenField(X, nominalFields);
      genY := NF2GenField(Y);
      myRF := int.RF_Regression(genX, genY, numTrees, featuresPerNode, maxDepth);
      model := myRF.GetModel;
      RETURN model;
    END;
    /**
      * Classify a set of data points using a previously fitted model
      *
      * @param X The set of independent data in NumericField format
      * @param mod A model previously returned by GetModel in Layout_Model2 format.
      * @return A NumericField dataset that provides a prediction for each X record.
      */
    EXPORT DATASET(NumericField) Predict(DATASET(NumericField) X, DATASET(Layout_Model2) mod) := FUNCTION
      genX := NF2GenField(X);
      myRF := int.RF_Regression();
      predictions := myRF.Predict(genX, mod);
      RETURN predictions;
    END;
  /**
    * Calculate R-squared, the Coefficient of Determination for the Regression
    *
    * R-squared varies from 0 to 1 (and in some cases negative).  R-squared of 1 indicates that the regression
    * explains all of the variance in the data, while zero or negative indicates that the mean is a better
    * predictor than the regression.
    * @param X the independent data  in NumericField format.
    * @param Y the dependent data in NumericField format.
    * @param mod the model to be tested as previously returned from GetModel(...).  In Layout_Model2 format.
    * @return a dataset indicating the R-squared for each work-item.
    **/
    EXPORT DATASET({UNSIGNED wi, REAL R2}) Rsquared(DATASET(NumericField) X,
                                                          DATASET(NumericField) Y,
                                                          DATASET(Layout_Model2) mod) := FUNCTION
      genX := NF2GenField(X);
      genY := NF2GenField(Y);
      myRF := int.RF_Regression();
      rsq := myRF.Rsquared(genX, genY, mod);
      RETURN rsq;
    END;
  END;
