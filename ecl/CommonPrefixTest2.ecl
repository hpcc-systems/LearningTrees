/*##############################################################################

    HPCC SYSTEMS software Copyright (C) 2022 HPCC Systems®.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
############################################################################## */

// Updated version of CommonPrefixLenTest that outputs whether the correct
// output is reached or what any differences were, plus additional inputs

IMPORT $.^ AS LT;
IMPORT LT.Internal AS int;

inp1 := [1, 2, 3, 4, 5];
inp2 := [1, 2, 4, 5, 6, 7];
inp3 := [1, 2, 3, 4, 5, 6, 7];
inp4 := [2, 3, 4, 5];
inp5 := [7, 9, 13, 20];
inp6 := [7, 9, 13, 20];

Res1x2 := int.CommonPrefixLen(inp1, inp2);
Res1x3 := int.CommonPrefixLen(inp1, inp3);
Res2x3 := int.CommonPrefixLen(inp2, inp3);
Res3x2 := int.CommonPrefixLen(inp3, inp2); // Test that function is symmetrix as f(2, 3) should equal f(3, 2) 
Res3x4 := int.CommonPrefixLen(inp3, inp4);
Res5x6 := int.CommonPrefixLen(inp5, inp6); // Equal sets, should return the length

Expected1x2 := 2;
Expected1x3 := 5;
Expected2x3 := 2;
Expected3x2 := 2;
Expected3x4 := 0;
Expected5x6 := 4;

Test_Result := RECORD
    STRING Test;
    INTEGER Expected;
    INTEGER Result;
END;

tests := DATASET([{'1x2', Expected1x2, Res1x2},
                  {'1x3', Expected1x3, Res1x3},
                  {'2x3', Expected2x3, Res2x3},
                  {'3x2', Expected3x2, Res3x2},
                  {'3x4', Expected3x4, Res3x4},
                  {'5x6', Expected5x6, Res5x6}], Test_Result);    

OUTPUT(IF(COUNT(tests(Expected != Result)) = 0, 'All Tests Passed', 'Test Cases Failed'), NAMED('Result'));
OUTPUT(tests(Expected != Result), NAMED('Errors'));