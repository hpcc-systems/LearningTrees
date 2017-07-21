IMPORT $.^.ndArray;

NumericArray := ndArray.NumericArray;

na := DATASET([
                  {1, 1.1, [1, 5, 1, 1]},
                  {1, 1.2, [1, 5, 1, 2]},
                  {1, 1.3, [1, 5, 1, 3]},
                  {1, 2.1, [1, 5, 2, 1]},
                  {1, 2.2, [1, 5, 2, 2]},
                  {1, 2.3, [1, 5, 2, 3]},
                  {1, .16, [1, 6, 1, 1]},
                  {1, .25, [2, 5, 1, 1]}
                  ], NumericArray);
rslt1 := ndArray.ToNumericField(na, [1,5]);

rslt2 := ndArray.FromNumericField(rslt1, [5,10]);

rslt3_1 := ndArray.GetItem(rslt2, [5, 10, 2, 3]);
rslt3_2 := ndArray.GetItem(rslt2, [5, 10, 2, 3], 2);  // Shouldn't find it -- no wi = 2

rslt4 := ndArray.SetItem(rslt2, 2.4, 1, [5, 10, 2, 4]);

rslt5 := ndArray.Extract(rslt4, [5, 10, 2]);

rslt6 := ndArray.Insert(na, rslt5, [1, 6, 1, 1]);

OUTPUT(rslt1, NAMED('ToNF'));
OUTPUT(rslt2, NAMED('FromNF'));
OUTPUT(rslt3_1, NAMED('GetItem'));
OUTPUT(rslt3_2, NAMED('GetItemNotFound'));
OUTPUT(rslt4, NAMED('SetItem'));
OUTPUT(rslt5, NAMED('Extract'));
OUTPUT(rslt6, NAMED('Insert'));
