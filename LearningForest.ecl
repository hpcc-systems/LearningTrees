/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2017 HPCC Systems®.  All rights reserved.
############################################################################## */

IMPORT $ AS LT;
IMPORT LT.LT_Types AS Types;
IMPORT ML_Core;
IMPORT ML_Core.Types as CTypes;
IMPORT $ as LT;
IMPORT LT.internal AS int;

GenField := Types.GenField;
DiscreteField := CTypes.DiscreteField;
NumericField := CTypes.NumericField;
Layout_Model2 := CTypes.Layout_Model2;
ModelStats := Types.ModelStats;
TreeNodeDat := Types.TreeNodeDat;

/**
  * This is the base module for Random Forests.
  * It implements the Random Forest algorithms as described by Breiman, 2001
  * (see https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
  *
  * @param numTrees The number of trees to create as the forest for each work-item.
  *                 This defaults to 100, which is adequate for most cases.
  * @param featuresPerNode The number of features to choose among at each split in
  *                 each tree.  This number of features will be chosen at random
  *                 from the full set of features.  The default is the square
  *                 root of the number of features provided, which works well
  *                 for most cases.
  * @param maxDepth The deepest to grow any tree in the forest.  The default is
  *                 100, which is adequate for most purposes.  Increasing this value
  *                 for very large and complex problems my provide slightly greater
  *                 accuracy at the expense of much greater runtime.
  */
  EXPORT LearningForest(UNSIGNED numTrees=100,
              UNSIGNED featuresPerNode=0,
              UNSIGNED maxDepth=100) := MODULE
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
      * Get summary statistical information about the model.
      *
      * @param mod A model previously returned from GetModel.
      * @return A single ModelStats record per work-item, containing information about the model
      *         for that work-item.
      * @see LT_Types.ModelStats
      */
    EXPORT DATASET(ModelStats) GetModelStats(DATASET(Layout_Model2) mod) := FUNCTION
      myRF := int.RF_Base();
      RETURN myRF.GetModelStats(mod);
    END;
    /**
      * Extract the set of tree nodes from a model.
      *
      * @param mod A model as returned from GetModel.
      * @return Set of tree nodes representing the fitted forest in
      *         DATASET(TreeNodeDat) format.
      * @see LT_Types.TreeNodeDat
      */
    EXPORT DATASET(TreeNodeDat) Model2Nodes(DATASET(Layout_Model2) mod) := FUNCTION
      myRF := int.RF_Base();
      nodes0 := myRF.Model2Nodes(mod);
      nodes := SORT(nodes0, wi, treeId, level, nodeId, LOCAL);
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
    myRF := int.RF_Base();
    fi := myRF.FeatureImportance(mod);
    RETURN fi;
  END;
  SHARED empty_data := DATASET([], NumericField);
  /**
    * <p>Calculate a matrix of distances between data points in Random Forest Decision Space (RFDS).
    * This is an experimental method and may not
    * scale to large numbers of data point combinations.
    * Two sets of data points X1 and X2 are taken as parameters.  A
    * Decision Distance will be returned for every point in X1 to every
    * point in X2.  Therefore, if X1 has N points and X2 has M points, an
    * N x M matrix of results will be produced.  X2 may be omitted, in which
    * case, an N x N matrix will be produced with a Decision Distance for
    * every pair of points in X1.
    *
    * <p>This metric represents a distance measure in the RFDS.
    * As such, it provides a continuous measure of distance in a space that is
    * highly non-linear and discontinuous relative to the training data.
    * Distances in RFDS can be thought of as the number of binary decisions
    * that separate two points in the tree.  DD, however is a normalized
    * metric 0 <= DD < 1 that incorporates the depth of the decision tree.
    * It is also averaged over all of the trees in the forest.
    * It can possibly be viewed as an approximation of the relative Hamming Distances
    * between points.
    *
    * @param mod The Random Forest model on which to base the distances.
    * @param X1 DATASET(NumericField) of "from" points.
    * @param X2 (Optional) DATASET(NumericField) of "to" points.  If this
    *             parameter is omitted, the X1 will be used as both "to" and
    *             "from" points.
    * @return DATASET(NumericField) matrix where 'id' is the id of the "from"
    *               point and 'number' is the id of the "to" point.
    *               'value' contains the DD metric between "from" and "to" points.
    *               Note that if the same point is in X1 and X2, there will be
    *               redundant metrics, since DD is a symmetric measure (i.e.
    *               DD(x1, x2) = DD(x2, x1).
    */
  EXPORT DecisionDistanceMatrix(DATASET(Layout_Model2) mod, DATASET(NumericField) X1,
                                  DATASET(NumericField) X2=empty_data) := FUNCTION
    myRF := int.RF_Base();
    ddm := myRF.DecisionDistanceMatrix(mod, NF2GenField(X1), NF2GenField(X2));
    RETURN DDM;
    END;
  /**
    *
    * Uniqueness Factor is an experimental metric that determines how far a given point
    * is (in Random Forest Decision Distance) from a set of other points.
    * It may not scale to large numbers of data points.
    *
    * Uniqueness Factor looks at the Decision Distance from each point to every other
    * point in a set.
    *
    * It is similar to Decision Distance (above), but rather than providing a distance of
    * each "from" point to every "to" point, it provides the average distance of each "from"
    * point to all of the "to" points.
    *
    * Like Decision Distance, UF lies on the interval: 0 <= UF < 1.
    *
    * A high value of UF may indicate an anomolous data point, while a low value may indicate
    * "typicalness" of a data point.  It may therefore have utility for anomaly detection
    * or conversely, for the identification of class prototypes (e.g. the members of a class
    * with the lowest UF).  In a two-step process one could potentially compute class prototypes
    * and then look at the distance of a point from all class prototypes.  This could result
    * in a way to detect anomalies with respect to e.g., known usage patterns.
    *
    * @param mod The Random Forest model on which to base the distances.
    * @param X1 DATASET(NumericField) of "from" points.
    * @param X2 (Optional) DATASET(NumericField) of "to" points.  If this
    *             parameter is omitted, the X1 will be used as both "to" and
    *             "from" points.
    * @return DATASET(NumericField) matrix where 'id' is the id of the "from"
    *               point and 'value' contains the UF metric for the point.
    *               I.e. the average DD of the "from" point to all "to" points.
    *               The 'number' field is not used and is set to 1.
    */
  EXPORT UniquenessFactor(DATASET(Layout_Model2) mod, DATASET(NumericField) X1, DATASET(NumericField) X2=empty_data) := FUNCTION
    myRF := int.RF_Base();
    uf := myRF.UniquenessFactor(mod, NF2GenField(X1), NF2GenField(X2));
    RETURN uf;
    END;

  /**
    * Compress and cleanup the model
    *
    * This function is provided to reduce the size of a model by compressing out
    * branches with only one child.  These branches are a result of the RF algorithm,
    * and do not affect the results of the model.
    * This is an expensive operation, which is why it is not done as a matter of
    * course.  It reduces the size of the model somewhat, and therefore slightly speeds
    * up any processing that uses the model, and reduces storage size.
    * You may want to compress the model if storage is at a premium, or if the model
    * is to be used many times (so that the slight performance gain is multiplied).
    * This also makes the model somewhat more readable, and could
    * be useful when analyzing the tree or converting it to another system
    * (e.g. LUCI) for processing.
    *
    * @param mod Model as returned from GetModel in Layout_Model2 format.
    * @return The Compressed Model.
    * @see ML_Core.Types.Layout_Model2
    *
    */
  EXPORT CompressModel(DATASET(Layout_Model2) mod) := FUNCTION
    myRF := int.RF_Base();
    cMod := myRF.CompressModel(mod);
    RETURN cMod;
  END;
END;
