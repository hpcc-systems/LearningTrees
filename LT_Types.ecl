/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2017 HPCC Systems®.  All rights reserved.
############################################################################## */
IMPORT $ AS LT;
IMPORT ML_core;
IMPORT ML_core.Types as CTypes;

t_Work_Item := CTypes.t_Work_Item;
t_Count := CTypes.t_Count;
t_RecordId := CTypes.t_RecordID;
t_FieldNumber := CTypes.t_FieldNumber;
t_FieldReal := CTypes.t_FieldReal;
t_Discrete := CTypes.t_Discrete;
t_TreeId := t_FieldNumber;
Layout_Model := CTypes.Layout_Model;
DiscreteField := CTypes.DiscreteField;
NumericField := CTypes.NumericField;
Field_Mapping := CTypes.Field_Mapping;
t_index := CTypes.t_index;

/**
  * Type definition module for Learning Trees.
  */
EXPORT LT_Types := MODULE
  /**
    * Type definition for the node id field representing a tree node's id
    */
  EXPORT t_NodeId := UNSIGNED8;
  /**
    * Definition of the meaning of the indexes of the Forest Model variables.
    * <p>Ind1 enumerates the first index, which
    * is used to determine which type of data is stored:<ul>
    * <li>nodes stores the list of tree nodes that describes the forest.
    *         The second index is just the sequential number of the node
    *         The third index is enumerated below (see Ind3_nodes).</li>
    * <li>samples stores the set of sample indexes (i.e. ids) associated
    *         with each treeId.
    *         The second index represents the treeId.  The third index
    *         represents the sample number. The value is the id of the
    *         sample in the original training dataset.
    *         {samples, treeId, sampleNum} -> origId.</li>
    * <li>classWeights (ClassificationForest only) stores the weights associated
    *         with each class label.  The second index represents the class
    *         label.  The value is the weight.  {classWeights, classLabel} -> weight.
    *         Class weights are only stored for Classification Forests.</li></ul>
    */
  EXPORT Forest_Model := MODULE
    /**
      * Index 1 represents the category of data within the model
      *
      * @value reserved = 1.  Reserved for future use.
      * @value nodes = 2.  The set of tree nodes within the model.
      * @value samples = 3. The particular record ids that are included in tree's sample .
      * @value classWeights = 4.  The weights assigned to each class (for ClassificationForest only).
      */
    EXPORT Ind1 := MODULE
      EXPORT t_index reserved := 1; // Reserved for future use
      EXPORT t_index nodes := 2;
      EXPORT t_index samples := 3;
      EXPORT t_index classWeights := 4;
    END;
    /**
      * For tree node data (i.e. Ind1 = nodes), the following
      * constant definitions are used for the different fields
      * of the tree-node.
      * Note that Ind1 indicates tree nodes, Ind2 represents the different nodes
      * and Ind3 defines the different fields.  For example, the treeId for the
      * first node would be stored at [2,1,1].  These correspond to the persisted
      * fields of TreeNodeDat with similar names.
      *
      * @value treeID = 1.  The tree identifier.
      * @value level = 2.  The level of the node within the tree.
      * @value nodeId = 3.  The nodeId of this node within the tree.
      * @value parentId = 4.  The parent node's nodeId.
      * @value isLeft = 5.  Left / Right indicator of this node within it's parent's chilren.
      * @value number = 6.  The field number to split on.
      * @value value = 7.  The value to compare against.
      * @value isOrd = 8.  Indicator of ordered vs categorical data.
      * @value depend = 9.  The value to predict for samples in this leaf.
      * @value support = 10.  The number of datapoints from the training data that reached
      *                       this node.
      * @value if = 11.  The 'impurity reduction' achieved by this branch.
      *
      */
    EXPORT Ind3_Nodes := MODULE
      EXPORT t_index treeId := 1;
      EXPORT t_index level := 2;
      EXPORT t_index nodeId := 3;
      EXPORT t_index parentId := 4;
      EXPORT t_index isLeft := 5;
      EXPORT t_index number := 6;
      EXPORT t_index value := 7;
      EXPORT t_index isOrd := 8;
      EXPORT t_index depend := 9;
      EXPORT t_index support := 10;
      EXPORT t_index ir := 11;
    END;
  END;

  /**
    * Definition of the meaning of the indexes of the Gradient Boosting Model variables.
    * <p>Ind1 enumerates the first index, which
    * is used to determine which type of data is stored:<ul>
    * <li>fModels stores the list of forest models that comprise the boosting
    *         hierarchy.  Each of these models can be decomposed by the Forest
    *         learning modules.</li>
    * <li>Other values are reserved for future use.
    */
  EXPORT Bf_Model := MODULE
    /**
      * Index 1 represents the category of data within the model
      *
      * @value reserved = 1.  Reserved for future use.
      * @value fModels = 2.  The set of forest models that comprise the boosting
      *                       hierarchy.
      */
    EXPORT Ind1 := MODULE
      EXPORT t_index reserved := 1; // Reserved for future use
      EXPORT t_index fModels := 2;
    END;
  END;

  /**
    * GenField extends NumericField by adding an isOrdinal field.  This
    * allows both Ordered and Nominal (Categorical) data to be held by the same record type.
    *
    * @field wi The work-item identifier for this cell.
    * @field id The record-identifier for this cell.
    * @field number The field number (i.e. featureId) of this cell.
    * @field value The numerical value of this cell.
    * @field isOrdinal TRUE if this field represents ordered data.  FALSE if it is categorical.
    * @see ML_Core.Types.NumericField.
    */
  EXPORT GenField := RECORD(NumericField)
    Boolean isOrdinal;
  END;


  /**
    * <p>This is the major working structure for building the forest.
    * <p>For efficiency and uniformity, this record structure serves several purposes
    * as the forest is built:
    * <ul><li>It represents all of the X,Y data associated with each tree and node as the
    *   forest is being built.  This case is recognized by id > 0 (i.e. it is a data point).
    *   wi, treeId, level, and NodeId represent the work-item and tree node with which the data is currently
    *         associated.
    *         All data in a tree's sample is originally assigned to the tree's root node (level = 1, nodeId = 1).
    *   <ul><li>id is the sample index in this trees data bootstrap sample.</li>
    *   <li>origId is the sample index in the original Independent(X) data.</li>
    *   <li>number is the field number from the X data.</li>
    *   <li>isOrdinal indicates whether this data is Ordinal (true) or Nominal (false).</li>
    *   <li>value is the data value of this data point.</li>
    *   <li>depend is the Dependent (Y) value associated with this data point.</li></ul></li>
    * <li>It represents the skeleton of the tree as the tree is built from the root down
    *   and the data points are subsumed (summarized) by the evolving tree structure.
    *   These cases can be identified by id = 0.<ul>
    *   <li>It represents branch (split) nodes:<ul>
    *       <li>id = 0 -- All data was subsumed.</li>
    *       <li>number > 0 -- The original field number of the Independent(X) variable on which to split.</li>
    *       <li>value -- the value on which to split</li>
    *       <li>parentId -- The nodeId of the branch at the previous level that leads to this
    *                   node.  Zero only for root.</li>
    *       <li>level -- The distance from the root (root = 1).</li>
    *       <li>support -- The number of data points that reach this node.</li>
    *       <li>ir -- The impurity reduction for this split.</li></ul></li>
    *   <li>It represents leaf nodes:<ul>
    *       <li>id = 0 -- All data was subsumed.</li>
    *       <li>number = 0 -- This discriminates a leaf from a branch node.</li>
    *       <li>depend has the Y value for that leaf.</li>
    *       <li>parentId has the nodeId of the branch node at the previous level.</li>
    *       <li>support has the count of samples that reached this leaf.</li>
    *       <li>level -- The depth of the node in the tree (root = 1).</li></ul></li></ul>
    * <p>Each tree starts with all sampled data points assigned to the root node (i.e. level = 1, nodeId = 1)
    * As the trees grow, data points are assigned to deeper branches, and eventually to leaf nodes, where
    * they are ultimately subsumed (summarized) and removed from the dataset.
    * <p>At the end of the forest growing process only the tree skeleton remains -- all the datapoints having
    * been summarized by the resulting branch and leaf nodes.
    * @field treeId The unique id of the tree in the forest.
    * @field nodeId The id of this node within the tree.
    * @field parentId The node id of this node's parent.
    * @field isLeft Indicates whether this node is the left child or the right child of the parent.
    * @field wi The work item with which this record is associated.
    * @field id The record id of the sample during tree construction.  Will be zero once the record has
    *           been replaced by a skeleton node (i.e. branch or leaf).
    * @field number The field number on which the branch splits
    * @field value The value of the data field, or the splitValue for a branch node.
    * @field level The level of the node within its tree.  Root is 1.
    * @field origId The sample index (id) of the original X data that this sample came from.
    * @field depend The dependent value associated with this id.
    * @field support The number of data samples subsumed by this node.
    * @field ir The 'impurity' reduction achieved by this branch.
    * @field observWeight The observation weight associated with this observation.
    */
  EXPORT TreeNodeDat := RECORD
    t_TreeID treeId;
    t_NodeID nodeId;
    t_NodeID parentId;
    BOOLEAN  isLeft;             // If true, this is the parent's left split
    GenField;                    // Instance Independent Data - one attribute
    UNSIGNED2     level;         // Level of the node in tree.  Root is 1.
    t_Discrete    origId;        // The sample index (id) of the original X data that this sample came from
    t_FieldReal   depend;        // Instance Dependent value
    t_RecordId   support:=0;    // Number of data samples subsumed by this node
    t_FieldReal  ir:=0;            // Impurity reduction at this node (branches only)
    t_FieldReal  observWeight:=1; // Weight assigned to this observation
  END;

  /**
    * Main data structure for processing Boosted Forest.
    * <p>The structure is the same as for random forests, but with an extra
    * field gbLevel that represents the level of the gradient boosted forest
    * nodes within the boosting hierarchy.
    * <p>Each set of nodes representing a forest is organized hierarchically based
    * on that field.
    * <p>Each level of the Boosted Forest contains a random forest.  The
    * results from each random forest are added together to get the final result
    * for the GBF.
    */
  EXPORT BfTreeNodeDat := RECORD(TreeNodeDat)
    UNSIGNED2 bfLevel;
  END;
  /**
    * The probability that a given sample is of a given class
    *
    * @field wi The work-item identifier.
    * @field id The record-id of the sample.
    * @field class The class label.
    * @field cnt The number of trees that predicted this class label.
    * @field prob The percentage of trees that assigned this class label,
    *             which is a rough stand-in for the probability that the label
    *             is correct.
    */
  EXPORT ClassProbs := RECORD
    t_Work_Item wi;  // Work-item id
    t_RecordID id;  // Sample identifier
    t_Discrete class; // The class label
    t_Discrete cnt; // The number of trees that assigned this class label
    t_FieldReal prob; // The percentage of trees that assigned this class label
                      // which is a rough stand-in for the probability that the
                      // label is correct.
  END;

  /**
    * NodeSummary provides information to identify a given tree node
    *
    * @field wi The work-item id for this node.
    * @field treeId The tree identifier within this work-item.
    * @field nodeId The node within the tree and work-item.
    * @field parentId The nodeId of this nodes parent node.
    * @field isLeft Boolean indicator of whether this is the Left child (TRUE) or
    *         Right child (FALSE) of the parent.
    * @field support The number of data samples that reached this node.
    */
  EXPORT NodeSummary := RECORD
    t_Work_Item wi;
    t_TreeID treeId;
    t_NodeID nodeId;
    t_NodeID parentId;     // Note that for any given (wi, treeId, nodeId), parentId and isLeft
                           //   will be constant, but we need to carry them through to maintain
                           //   the integrity of the nodes' relationships.
    BOOLEAN isLeft:=True;
    t_RecordId support;   // The number of data samples reaching this node.
  END;
  /**
    * SplitDat is used to hold information about a potential split.
    * It is based on the NodeSummary record type above.  It adds the following fields
    *
    * @field number The field number of the Independent data that is being used to split.
    * @field splitVal The value by which to split the data.
    * @field isOrdinal TRUE indicates that it is an ordered value and will use a
    *                  greater-than-or-equal split (i.e. value >= splitVal).
    *                  FALSE indicates that the values are nominal
    *                  (i.e. categorical) and will use an equal-to split (i.e. value = splitVal)
    */
  EXPORT SplitDat := RECORD(NodeSummary)
    t_FieldNumber number;  // This is the field number that is being split
    t_FieldReal splitVal;  // This is the value at which to split <= splitval => LEFT >splitval
                           // => right
    BOOLEAN isOrdinal;     // We need to carry this along
    t_FieldReal ir;        // Impurity reduction at this split
  END;

  /**
    * NodeImpurity carries identifying information for a node as well as its impurity level
    * It is based on the NodeSummary record type above, but includes an assessment of the
    * 'impurity' of the data at this node (i.e. GINI, Variance, Entropy).
    *
    * @field impurity The level of impurity at the given node.  Zero is most pure.
    */
  EXPORT NodeImpurity := RECORD(NodeSummary)
    t_FieldReal impurity;  // The level of impurity of the given node.  Zero is most pure.
  END;

  /**
    * Provides a summary of each work item for use in building the forest.
    *
    * @field wi The work-item identifier.
    * @field numSamples The number of samples within this work-item
    * @field numFeatures The number of features (i.e. number fields in the Independent
    *                    data for this work-item.
    * @field featuresPerNode The number of features to be randomly chosen at each level
    *                         of tree building.  It is a function of, the user parameter
    *                         'featuresPerNode' and the number of features in the work-item
    *                         numFeatures.
    */
  EXPORT wiInfo := RECORD
    t_Work_Item   wi;               // Work-item Id
    t_RecordId    numSamples;       // Number of samples for this wi's data
    t_FieldNumber numFeatures;      // Number of features for this wi's data
    t_Count       featuresPerNode;  // Features per node may be different for each work-item
                                    //   because it is based on numFeatures as well as the
                                    //   featuresPerNodeIn parameter to the module.
  END;
  /**
    * Model Statistics Record
    *
    * Provides descriptive information about a Model
    *
    * @field wi The work-item whose model is described
    * @field treeCount The number of trees in the forest
    * @field minTreeDepth The depth of the shallowest tree
    * @field maxTreeDepth The depth of the deepest tree
    * @field avgTreeDepth The average depth of all trees
    * @field minTreeNodes The number of nodes in the smallest tree
    * @field maxTreeNodes The number of nodes in the biggest tree
    * @field avgTreeNodes The average number of nodes for all trees
    * @field totalNodes The number of nodes in the forest
    * @field minSupport The minimum sum of support for all trees.
    *                   Support indicates the number of training datapoints
    *                   that arrived at a given leaf node
    * @field maxSupport The maximum sum of support for all trees
    * @field agvSupport The average sum of support for all trees
    * @field avgSupportPerLeaf The average number of data points per
    *                     leaf across the forest
    * @field maxSupportPerLeaf The maximum data points at any single
    *                     leaf across the forest
    * @field avgLeafDepth The average depth for all leaf nodes
    *                     for all trees
    * @field minLeafDepth The minimum depth for all leaf nodes
    *                     for all trees
    */
  EXPORT ModelStats := RECORD
    t_Work_Item wi;
    UNSIGNED treeCount;
    UNSIGNED minTreeDepth;
    UNSIGNED maxTreeDepth;
    REAL avgTreeDepth;
    UNSIGNED minTreeNodes;
    UNSIGNED maxTreeNodes;
    REAL avgTreeNodes;
    UNSIGNED totalNodes;
    UNSIGNED minSupport;
    UNSIGNED maxSupport;
    REAL avgSupport;
    REAL avgSupportPerLeaf;
    UNSIGNED maxSupportPerLeaf;
    REAL avgLeafDepth;
    UNSIGNED minLeafDepth;
    UNSIGNED bfLevel := 1;
  END; // ModelStats
  /**
    * Feature Importance Record
    * describes the importance of each feature.
    * @field wi The work-item associated with this information.
    * @field number The feature number.
    * @field importance The 'importance' metric.  Higher value is more
    *                   important.
    * @field uses The number of times the feature was used in the forest.
    */
  EXPORT FeatureImportanceRec := RECORD
    t_Work_Item wi;
    t_FieldNumber number;
    t_FieldReal importance;
    UNSIGNED uses;
  END;

  /**
    * ClassWeightsRecord holds the weights associated with each
    * class label.
    *
    * @field wi The work-item.
    * @field classLabel The subject class label.
    * @field weight The weight associated with this class label.
    **/
  EXPORT ClassWeightsRec := RECORD
    t_work_item wi;
    t_Discrete classLabel;
    t_FieldReal weight;
  END;

  /**
    * Structure used to describe the Scorecards for LUCI format export.
    *
    * For a single scorecard model, a single LUCI_Scorecard record is used.
    * For multiple scorecards, one record is required per scorecard.
    * One L2SC or L2FO record will be generated per scorecard, and additionally
    * One L2SE record will be generated for each scorecard with a non-blank
    * 'filter_expr'.
    *
    * @field wi_num The work-item number on which to base this scorecard or '1' if only one
    *               work-item / scorecard us used.
    * @field scorecard_name The LUCI name for this scorecard.
    * @field filter_expr Optional -- An expression on the LUCI input dataset layout that selects
    *                    the records to be included in this scorecard (e.g. 'state_id = 2').
    *                    If the expression contains strings, the single-quotes must be preceded
    *                    by a backslash escape character (e.g. 'state = \'NY\'').
    *                    The filter expression must follow ECL Boolean expression syntax.
    *                    It should be blank if all records are to be used.  See L2SE LUCI
    *                    record format, Scorecard-Election-Criteria for more details.
    * @field fieldMap    A DATASET(Field_Mapping) as returned from the FromField macro that maps the Field Names
    *                    (as used in the LUCI definition) to the field numbers (as used in the ML model).
    *                    Note: must be the same set of fields used in training the forest for this work item.
    */
  EXPORT LUCI_Scorecard := RECORD
    UNSIGNED wi_num := 1;
    STRING scorecard_name;
    STRING filter_expr := '';
    DATASET(Field_Mapping) fieldMap;
  END;
END; // LT_Types
