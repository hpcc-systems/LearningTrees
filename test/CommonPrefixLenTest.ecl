/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2017 HPCC Systems®.  All rights reserved.
############################################################################## */

/**
  * Test the CommonPrefixLen C++ method
  */

IMPORT $.^ AS LT;
IMPORT LT.Internal AS int;


s1 := [1, 2, 3, 4, 5];
s2 := [1, 2, 4, 5, 6, 7];
s3 := [1, 2, 3, 4, 5, 6, 7];
s4 := [2, 3, 4, 5];

cpl1 := int.CommonPrefixLen(s1, s2);

OUTPUT(cpl1, NAMED('s1s2'));

cpl2 := int.CommonPrefixLen(s1, s3);

OUTPUT(cpl2, NAMED('s1s3'));

cpl3 := int.CommonPrefixLen(s3, s2);
OUTPUT(cpl3, NAMED('s3s2'));

cpl4 := int.CommonPrefixLen(s2, s3);
OUTPUT(cpl4, NAMED('s2s3'));

cpl5 := int.CommonPrefixLen(s3, s4);
OUTPUT(cpl5, NAMED('s3s4'));
