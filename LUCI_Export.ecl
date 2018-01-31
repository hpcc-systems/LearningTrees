/**
  * Export a Learning Forest model to LUCI format
  *
  * LUCI is a LexisNexis proprietary mechanism for describing a model that can then
  * be efficiently processed within an LN product.
  *
  * Note the following restrictions:
  * 1) Regression Forest Only.  Classification Forests cannot be handle by LUCI at this
  *    time because there is no support for aggregation by vote (mode).
  * 2) This module produces a LUCI file that outputs the exact same Regression values
  *    as the Regression Forest model.  LUCI allows some additional features that are
  *    beyond the scope of this module.
  *    If these features are needed, the resultant LUCI file may need to be hand edited
  *    to achieve those results.  Examples of these features include:
  *    - Defining Reason Code logic (L1MD record)
  *    - Setting minimum and maximum bounds (L2FO record)
  *    - Adding an increment value to the final results (L2FO)
  *    - Setting a scaling formula to scale the final results (L2FO)
  *    - Excluding certain input records (L1EX record)
  *    - See the LUCI documentation for more info:
  *          https://gitlab.ins.risk.regn.net/HIPIE/HIPIE_Plugins/wikis/LUCIfiles#l2se
  * This module supports the following LUCI use cases:
  * 1)  Single work-item / single scorecard.
  * 2)  Work-items and corresponding scorecards represent training of different response
  *     variables on (potentially) different subsets of the features in the LUCI input layout.
  * 3)  Work-items and corresponding scorecards represent training of the same response variable
  *     across subsets of the input data (e.g. one per country).  It is anticipated, though not
  *     required, that the same subset of LUCI input layout features was used for training each subset.
  *
  */
IMPORT $ AS LT;
IMPORT LT.LT_Types as Types;
IMPORT Std.Str;
IMPORT Std.System.ThorLib;
IMPORT ML_Core;
IMPORT ML_Core.Types as CTypes;
IMPORT ML_Core.ModelOps2;

Layout_Model2 := CTypes.Layout_Model2;
LUCI_Rec := CTypes.LUCI_Rec;
LUCI_Scorecard := Types.LUCI_Scorecard;
t_FieldNumber := CTypes.t_FieldNumber;
FM := Types.Forest_Model;
FM1 := FM.Ind1; // The first index of the model

TreeNodeDat := Types.TreeNodeDat;

/**
  * Export a LUCI model
  *
  * Create a LUCI model based on a model as returned from
  * RegressionForest.GetModel, as well as additional information provided
  * within this interface.
  * The following types of LUCI record are created:
  * - A single L1MD record
  * - One L2FO record per LUCI scorecard
  * - One L2SE record per scorecard that includes a filter expression
  * - One L3TN record per node for each tree in each forest (i.e. work-item).
  * Note that scorecards in LUCI correspond to work-items in LearningForest.
  *
  * @param mod The random forest model as returned from GetModel
  * @param model_name The name of the LUCI model (see LUCI L1MD definition)
  * @param model_id The id of the LUCI model (see LUCI L1MD definition)
  * @param scorecards DATASET(LT_Types.LUCI_Scorecard) describing each
  *                   work-item in the model that will be exported as
  *                   a LUCI scorecard.
  * @return DATASET(LUCI_Rec) representing the lines of a LUCI .csv file.
  *         The caller is responsible for melding the lines into an actual
  *         .csv file and storing it in a given location.
  */
EXPORT DATASET(LUCI_Rec)
      LUCI_Export(DATASET(Layout_Model2) mod,
                 STRING model_id,
                 STRING model_name,
                 DATASET(LUCI_Scorecard) scorecards) := FUNCTION
  myLF := LT.LearningForest();
  // Determine whether this is a Classification or Regression model.
  // If it has class weights, it is Classification.
  modCW := ModelOps2.Extract(mod, [FM1.classWeights]);
  isClassification := COUNT(modCW) > 0;
  aggregationType := IF(isClassification, 'MODE', 'AVE');
  // We're going to create a series of LUCI .csv records.
  // First we'll create the L1MD record
  model_type := IF(COUNT(scorecards) > 1, 'multi', 'single');
  L1MD := DATASET(['L1MD,' + model_id + ',' + model_name + ',' + model_type + ',,LT,0'], LUCI_rec);
  // Now create the L2FO record for each scorecard, and the L2SE record for any scorecard
  // with a filter expression.
  LUCI_rec make_L2FO(LUCI_Scorecard sc) := TRANSFORM
    SELF.line := 'L2FO,' + model_id + ',' + sc.scorecard_name + ',' + aggregationType + ',0,,N,N,,N,';
  END;
  L2FO := PROJECT(scorecards, make_L2FO(LEFT));
  LUCI_rec make_L2SE(LUCI_Scorecard sc) := TRANSFORM
    SELF.line := 'L2SE,' + model_id + ',' + sc.scorecard_name + ',"' + sc.filter_expr + '"';
  END;
  L2SE := PROJECT(scorecards(filter_expr != ''), make_L2SE(LEFT));
  // Now create the L3TN records, one per tree node, which is the meat of the model.
  // First compress the model to remove any single-child splits
  cMod := myLF.CompressModel(mod);
  // Transform the TreeNodes to a form compatible with LUCI
  // In the model, the tree nodes each contain a parentId and a isLeft indicator.
  // In a LUCI model, the tree nodes contain a leftChildId and rightChildId.
  nodes := myLF.model2nodes(cMod);
  nodesD := SORT(DISTRIBUTE(nodes, HASH32(wi, treeId)), wi, treeId, level, nodeId, LOCAL);
  // First, extend the tree-nodes by adding a globally unique node identifier (i.e. unique within a tree).
  nodesG := GROUP(nodesD, wi, treeId, LOCAL);
  eNodesG := PROJECT(nodesG, TRANSFORM({UNSIGNED gnid, TreeNodeDat}, SELF.gnid := COUNTER, SELF := LEFT), LOCAL);
  eNodes := UNGROUP(eNodesG);
  // Now add the right and left child gnids to the parent record.
  transNodes0 := JOIN(eNodes, eNodes(isLeft = TRUE), LEFT.wi = RIGHT.wi AND
                                      LEFT.treeId = RIGHT.treeId AND
                                      LEFT.level = RIGHT.level-1 AND
                                      LEFT.nodeId = RIGHT.parentId,
                                    TRANSFORM({eNodes, UNSIGNED leftChild, UNSIGNED rightChild},
                                              SELF.leftChild := RIGHT.gnid, SELF.rightChild := 0, SELF := LEFT),
                                    LEFT OUTER, LOCAL);
  transNodes1 := JOIN(transNodes0, transNodes0(isLeft = FALSE), LEFT.wi = RIGHT.wi AND
                                      LEFT.treeId = RIGHT.treeId AND
                                      LEFT.level = RIGHT.level-1 AND
                                      LEFT.nodeId = RIGHT.parentId,
                                    TRANSFORM({transNodes0},
                                              SELF.rightChild := RIGHT.gnid, SELF := LEFT),
                                    LEFT OUTER, LOCAL);
  transNodes := SORT(transNodes1, wi, treeId, gnid);

  LUCI_rec make_L3TN({transNodes} tn, LUCI_Scorecard sc) := TRANSFORM
    fmap := sc.fieldMap;
    GetFieldName(t_FieldNumber fnum) := FUNCTION
      fName := fmap(assigned_name=(STRING)fnum)[1].orig_name;
      return fName;
    END;
    SELF.line := 'L3TN,' + model_id + ',' + sc.scorecard_name + ',' + tn.treeId + ',' + tn.gnid + ',' +
                    IF(tn.number > 0, GetFieldName(tn.number), '-1') + ',' + IF(tn.number > 0, tn.value, tn.depend) +
                    ',' + tn.leftChild + ',' + tn.rightChild + ',0,' + IF(tn.isOrdinal, 'LE', 'E');
  END;

  L3TN := JOIN(transNodes, scorecards, LEFT.wi = RIGHT.wi_num, make_L3TN(LEFT, RIGHT), LOOKUP);
  // Combine all the different record types into a single dataset.
  allRecs := L1MD & L2FO & L2SE & L3TN;
  RETURN allRecs;
END;
