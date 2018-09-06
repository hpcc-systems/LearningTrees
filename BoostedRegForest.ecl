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
GenField := Types.GenField;
ModelStats := Types.ModelStats;
BfTreeNodeDat := Types.BfTreeNodeDat;
Layout_Model2 := CTypes.Layout_Model2;
IRegression2 := Interfaces.IRegression2;
/**
  * Regression using Boosted Forests.
  *
  * <p>Boosted Forests (BF) are a combination of Gradient Boosted Trees (GBT) and
  * Random Forests (RF).  They provide accuracy at least as good as GBTs with the
  * ease of use of RFs.  They utilize Boosting to enhance the
  * accuracy of Random Forests.  Layers of Random Forests are constructed, each
  * attempting to compensate for the cumulative error of the forests before it.
  *
  * <p>A Boosted Forest with a forest size of 1 is
  * essentially the same as a GBT.  While this is supported, it is not recommended.
  * BFs with forest size >= 10 have characteristics superior to both RF and GBT.
  * They generally provide accuracy higher than RFs, and better than or equal that
  * of expertly regularized GBTs.  Yet they require no regularization, work well
  * with default parameters, and are as easy to use as RFs.
  *
  * <p>BFs provide an early-stopping capability.  This allows the number of boosting
  * iterations (i.e. maxLevels) to be specified at a high level (default 999),
  * but only boosts for as many iterations as necessary to maximize accuracy.
  * In normal practice, boosting will stop long before the default maxLevels is
  * reached.
  *
  * <p>Boosted Forests share most of the benefits and limitations of Random Forests:<ul>
  * <li>Random Forests provide an effective method for regression.  They are known
  * to be one of the best out-of-the-box methods as there are few assumptions
  * made regarding the nature of the data or its relationships.</li>
  * <li>Random Forests can effectively manage large numbers
  * of features, and will automatically choose the most relevant features.</li>
  * <li>Regression Forests can handle non-linear and discontinuous relationships
  * among features.</li>
  * <li>A limitation of Regression Forests is that they provide no extrapolation
  * beyond the bounds of the training data.  The training set should extend to the
  * limits of expected feature values.</li></ul>
  *
  * <p>This implementation allows both Ordinal (discrete or continuous) and
  * Nominal (unordered categorical values) for the independent (X) features.
  * There is therefore, no need to one-hot encode categorical features.
  * Nominal features should be identified by including their feature 'number'
  * in the set of 'nominalFields'.
  *
  * <p>Boosted Forests support the Myriad interface meaning that multiple
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
  * <p>While we recommend use of the default training parameters, we do allow override of
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
  * parameters have minimal effect, though slight gains may be possible in certain
  * circumstances.</li>
  * <li>For GBT (forestSize = 1), smaller sizes of maxTreeDepth (3-10) are recommended,
  * as are low values for learningRate (< .5).</li>
  * <li>For BF (forestSize > 9), moderate values are likely to provide optimal results:
  * maxTreeDepth between 7 and 30 and learningRate between .5 and 1.0 might generate
  * good results.</li></ul>
  *
  *
  * @param maxLevels The maximum number of boosting iterations to perform.  This is
  *         overridden by early stopping, and is primarily a failsafe in case the
  *         data is non-separable.  Default (recommended) is 999.
  * @param forestSize The number of trees to use in each Random Forest level.  The
  *         default (recommended) is zero, which indicates that it should be
  *         automatically determined by the software.
  * @param maxTreeDepth The depth to which trees are grown.  Smaller numbers provide
  *         weaker learners that are needed for GBT purposes.  The default (recommended)
  *         is 20.
  * @param learningRate The distance along the gradient to proceed on each boosting
  *         iteration.  The default (recommended) is 1.0.
  * @param earlyStopThreshold A threshold against the RVR (Residual Variance Ratio) to
  *         enable early stopping.  The default threshold (recommended) is .0001, which
  *         indicates that we will stop when 99.99% of the variance in the original
  *         data has been explained by the model.  Setting this value to zero disables
  *         early stopping (not recommended).
  *
  * @param nominalFields An optional set of field 'numbers' that represent Nominal (i.e. unordered,
  *                      categorical) values.  Specifying the nominal fields improves run-time
  *                      performance on these fields and often improves accuracy as well.  Binary fields
  *                      (fields with only two values) need not be included here as they can be
  *                      considered either ordinal or nominal.  The default is to treat all fields as
  *                      ordered.  Note that this feature should only be used if all of the independent
  *                      data for all work-items use the same record format, and therefore have the same
  *                      set of nominal fields.
  */
  EXPORT BoostedRegForest(
                          UNSIGNED maxLevels=255,
                          UNSIGNED forestSize=0,
                          UNSIGNED maxTreeDepth=20,
                          REAL8 learningRate=1.0,
                          REAL8 earlyStopThreshold=.0001,
                          SET OF UNSIGNED nominalFields=[]) := MODULE(IRegression2)
    // Map a NumericField dataset to GenField dataset
    SHARED DATASET(GenField) NF2GenField(DATASET(NumericField) ds, SET OF UNSIGNED nominalFields=[]) := FUNCTION
      dsOut := PROJECT(ds, TRANSFORM(GenField, SELF.isOrdinal := LEFT.number NOT IN nominalFields, SELF := LEFT));
      RETURN dsOut;
    END;
    // Map a DiscreteField dataset to GenField dataset
    SHARED DATASET(GenField) DF2GenField(DATASET(DiscreteField) ds) := FUNCTION
      dsOut := PROJECT(ds, TRANSFORM(GenField, SELF.isOrdinal := TRUE, SELF := LEFT));
      RETURN dsOut;
    END;
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
      myBF := int.BF_Regression(genX, genY, maxLevels, forestSize, maxTreeDepth, learningRate, earlyStopThreshold);
      model := myBF.GetModel;
      RETURN model;
    END;
    /**
      * Predict a set of data points using a previously fitted model.
      *
      * @param mod A model previously returned by GetModel in Layout_Model2 format.
      * @param observations The set of independent data in NumericField format.
      * @return A NumericField dataset that provides a prediction for each record in observations.
      */
    EXPORT DATASET(NumericField) Predict(DATASET(Layout_Model2) model, DATASET(NumericField) observations) := FUNCTION
      myBF := int.BF_Regression();
      predictions := myBF.Predict(model, observations);
      RETURN predictions;
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
      myBF := int.BF_Regression();
      RETURN myBF.GetModelStats(mod);
    END;
    /**
      * Extract the set of tree nodes from a model.
      *
      * @param mod A model as returned from GetModel.
      * @return Set of tree nodes representing the fitted forest in
      *         DATASET(TreeNodeDat) format.
      * @see LT_Types.TreeNodeDat
      */
    EXPORT DATASET(BfTreeNodeDat) Model2Nodes(DATASET(Layout_Model2) mod) := FUNCTION
      myBF := int.BF_Base();
      nodes0 := myBF.Model2Nodes(mod);
      nodes := SORT(nodes0, wi, bfLevel, treeId, level, nodeId, LOCAL);
      RETURN nodes;
    END;
  /**
    * <p>Determine the relative importance of features in the decision process of
    * the model.
    * Calculate feature importance using the Mean Decrease Impurity (MDI) method
    * from "Understanding Random Forests: by Gilles Loupe (https://arxiv.org/pdf/1407.7502.pdf)
    * and due to Breiman [2001, 2002].
    *
    * <p>Each feature is ranked by:
    * <pre>  SUM for each branch node in which feature appears (across all trees):
    *     (impurity_reduction * number of nodes split) / numTrees.</pre>
    * @param mod The model to use for ranking of feature importance.
    * @return DATASET(FeatureImportanceRec), one per feature per wi.
    * @see LT_Types.FeatureImportanceRec
    */
  EXPORT FeatureImportance(DATASET(Layout_Model2) mod) := FUNCTION
    myBF := int.BF_Base();
    fi := myBF.FeatureImportance(mod);
    RETURN fi;
  END;
END;
