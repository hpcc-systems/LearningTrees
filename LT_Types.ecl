/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2017 HPCC Systems®.  All rights reserved.
############################################################################## */
IMPORT $ AS LT;
IMPORT ML_core;
IMPORT ML_core.Types as CTypes;
IMPORT LT.ndArray;
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
NumericArray := ndArray.NumericArray;
t_index := ndArray.t_index;

/**
  * Type definition module for Learning Trees.
  */
EXPORT LT_Types := MODULE

  EXPORT t_NodeId := t_FieldNumber;

  /**
    * New Model Layout.  Should eventually be moved to ML_Core.
    *
    * Model is stored as an N-Dimensional numeric array via the
    * ndArray.NumericArray class.
    */
  EXPORT Layout_Model2 := NumericArray;

  /**
    * Definition of the meaning of the indexes of the Model's
    * NumericArray.  Ind1 enumerates the first index, which
    * is used to determine which type of data is stored.
    * - nodes stores the list of tree nodes that describes the forest.
    *         The second index is just the sequential number of the node
    *         The third index is enumerated below (Ind3_nodes).
    * - samples stores the set of sample indexes (i.e. ids) associated
    *         with each treeId.
    *         The second index represents the treeId.  The third index
    *         represents the sample number. The value is the id of the
    *         sample in the original training dataset.
    *         [<samples, <treeId>, <sampleNum>] -> origId
    * - classWeights (ClassificationForest only) stores the weights associated
    *         with each class label.  The second index represents the class
    *         label.  The value is the weight.  [<classWeights>, <classLabel>] -> weight
    *         Class weights are only stored for Classification Forests.
    */
  EXPORT Forest_Model := MODULE
    /**
      * Index 1 represents the category of data within the model
      */
    EXPORT Ind1 := MODULE
      EXPORT t_index nodes := 1;
      EXPORT t_index samples := 2;
      EXPORT t_index classWeights := 3;
    END;
    /**
      * For tree node data (i.e. Ind1 = nodes), the following
      * constant definitions are used for the different fields
      * of the tree-node.
      * Note that Ind1 indicates tree nodes, Ind2 represents the different nodes
      * and Ind3 defines the different fields.  For example, the treeId for the
      * first node would be stored at [1,1,1].  These correspond to the persisted
      * fields of TreeNodeDat with similar names.
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
    * GenField extends NumericField by adding an isOrdinal field.  This
    * allows both Ordered and Nominal (Categorical) data to be held by the same record type.
    *
    */
  EXPORT GenField := RECORD(NumericField)
    Boolean isOrdinal;
  END;

  /**
    * Tree node data
    * This is the major working structure for building the forest.
    * For efficiency and uniformity, this record structure serves several purposes
    * as the forest is built:
    * 1) It represents all of the X,Y data associated with each tree and node as the
    *   forest is being built.  This case is recognized by id > 0 (i.e. it is a data point)
    *   wi, treeId, level, and NodeId represent the work-item and tree node with which the data is currently
    *         associated.
    *         All data in a trees sample is originally assigned to the root node (level = 1, nodeId = 1)
    *         of its associated tree.
    *   id is the sample index in this trees data bootstrap sample
    *   origId is the sample index in the original X data.
    *   number is the field number from the X data
    *   isOrdinal indicates whether this data is Ordinal (true) or Nominal (false)
    *   value is the data value of this data point
    *   depend is the Y (dependent) value associated with this data point
    * 2) It represents the skeleton of the tree as the tree is built from the root down
    *   and the data points are subsumed (summarized) by the evolving tree structure.
    *   These cases can be identified by id = 0.
    *   2a) It represents branch (split) nodes:
    *       id = 0 -- All data was subsumed
    *       number > 0 -- The original field number of the X variable on which to split
    *       value -- the value on which to split
    *       parentId -- The nodeId of the branch at the previous level that leads to this
    *                   node.  Zero only for root.
    *       level -- The distance from the root (root = 1)
    *       support -- The number of data points that reach this node
    *       ir -- The impurity reduction for this split
    *   2b) It represents leaf nodes:
    *       id = 0 -- All data was subsumed
    *       number = 0 -- This discriminates a leaf from a branch node
    *       depend has the Y value for that leaf
    *       parentId has the nodeId of the branch node at the previous level
    *       support has the count of samples that reached this leaf
    *       level -- The depth of the node in the tree (root = 1)
    * Each tree starts with all sampled data points assigned to the root node (i.e. level = 1, nodeId = 1)
    * As the trees grow, data points are assigned to deeper branches, and eventually to leaf nodes, where
    * they are ultimately subsumed (summarized) and removed from the dataset.
    * At the end of the forest growing process only the tree skeleton remains -- all the datapoints having
    * been summarized by the resulting branch and leaf nodes.
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
    t_FieldReal  ir;            // Impurity reduction at this node (branches only)
  END;

  /**
    * ClassProbs represent the probability that a given sample is of a given class
    *
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
    * SplitDat is used to hold information about a potential split
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
    */
  EXPORT NodeImpurity := RECORD(NodeSummary)
    t_FieldReal impurity;  // The level of impurity of the given node.  Zero is most pure.
  END;

  /**
    * wiInfo provides a summary of each work item
    */
  EXPORT wiInfo := RECORD
    t_Work_Item   wi;               // Work-item Id
    t_RecordId    numSamples;       // Number of samples for this wi's data
    t_FieldNumber numFeatures;      // Number of features for this wi's data
    t_Count       featuresPerNode;  // Features per node may be different for each work-item
                                    //   because it is base on numFeatures as well as the
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
    * @field avgSupportPerLeaf The average number of datapoints per
    *                     leaf across the forest
    * @field maxSupportPerLeaf The maximum datapoints at any single
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
  END;
  /**
    * Feature Importance Record
    *
    * Describes the importance of each feature
    * @field wi The work-item associated with this information
    * @field number The feature number
    * @field importance The 'importance' metric.  Higher value is more
    *                   important.
    * @field uses The number of times the feature was used in the forest
    */
  EXPORT FeatureImportanceRec := RECORD
    t_Work_Item wi;
    t_FieldNumber number;
    t_FieldReal importance;
    UNSIGNED uses;
  END;
END; // LT_Types
