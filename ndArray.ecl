IMPORT ML_Core.Types;

t_work_item := Types.t_work_item;
t_fieldReal := Types.t_fieldReal;
t_index := UNSIGNED4;
t_indexes := SET OF t_index;
NumericField := Types.NumericField;

EXPORT ndArray := MODULE
  /**
    * Record format for Numeric ndArray
    */
  EXPORT NumericArray := RECORD
    t_work_item wi;
    t_fieldReal value;
    t_indexes indexes;
  END;

  SHARED empty_array := DATASET([], NumericArray);
  /**
    * Extract an inner ndArray (sub-array) from an existing ndArray
    * Work-item = 0 (default) will extract all work-items
    */
  EXPORT DATASET(NumericArray) Extract(DATASET(NumericArray) tens,
                                       t_indexes fromIndx, t_work_item fromWi=0) := FUNCTION
    NumericArray extract_indexes(NumericArray t, UNSIGNED prefixSize) := TRANSFORM
      outIndex := t.indexes[prefixSize+1.. ];
      SELF.indexes := outIndex;
      SELF         := t;
    END;
    prefixSize := COUNT(fromIndx);
    indxFilter := tens.indexes[..prefixSize] = fromIndx;
    filter     := IF(fromWi = 0, indxFilter, indxFilter AND tens.wi = fromWi);
    outTens    := PROJECT(tens(filter), extract_indexes(LEFT, prefixSize));
    return outTens;
  END;
  /**
    * Insert an ndArray (sub-array) into an existing ndArray
    */
  EXPORT DATASET(NumericArray) Insert(DATASET(NumericArray) arr1, DATASET(NumericArray) arr2, t_indexes atIndx) := FUNCTION
    NumericArray extend_indexes(NumericArray t) := TRANSFORM
      indxs := atIndx + t.indexes;
      SELF.indexes := indxs;
      SELF         := t;
    END;
    arr2a := PROJECT(arr2, extend_indexes(LEFT));
    return arr1 + arr2a;
  END;
  /**
    * Convert a 2D ndArray into a NumericField dataset
    */
  EXPORT ToNumericField(DATASET(NumericArray) arr, t_indexes fromIndx = []) := FUNCTION
    NumericField array_to_nf(NumericArray t) := TRANSFORM
      prefixSize := COUNT(fromIndx);
      suffix := t.indexes[prefixSize+1.. ];
      SELF.id := suffix[1];
      SELF.number := suffix[2];
      SELF := t;
    END;
    prefixSize := COUNT(fromIndx);
    filter := arr.indexes[..prefixSize] = fromIndx;
    outNF := PROJECT(arr(filter), array_to_nf(LEFT));
    return outNF;
  END;
  /**
    * Convert a NumericField dataset to a 2 dimensional ndArray
    *
    */
  EXPORT FromNumericField(DATASET(NumericField) nf, t_indexes atIndex=[]) := FUNCTION
    NumericArray nf_to_array(NumericField n) := TRANSFORM
      indexes := atIndex + [n.id, n.number];
      SELF.indexes := indexes;
      SELF         := n;
    END;
    outArr := PROJECT(nf, nf_to_array(LEFT));
    return outArr;
  END;
  /**
    * Get a single record (cell) from an ndArray using a set of coordinates
    */
  EXPORT GetItem(DATASET(NumericArray) arr, t_indexes indxs, wi_num=1) := FUNCTION
    return arr(indexes=indxs AND wi=wi_num)[1];
  END;
  /**
    * Add a single record (cell) to an ndArray at a given set of coordinates
    */
  EXPORT SetItem(DATASET(NumericArray) arr, t_fieldReal value, t_work_item wi, t_indexes indexes) := FUNCTION
    return arr + DATASET([{wi, value, indexes}], NumericArray);
  END;
END;
