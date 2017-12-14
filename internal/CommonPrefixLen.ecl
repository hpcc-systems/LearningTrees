/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2017 HPCC Systems®.  All rights reserved.
############################################################################## */
/**
  * Return the the length of the longest common prefix between two id-lists
  *
  * General Function on a list (i.e. ordered set)
  *
  * @param s1 The first list (SET OF UNSIGNED4)
  * @param s2 The second list (SET OF UNSIGNED4)
  *
  * @return The prefix size
  */
EXPORT UNSIGNED4 CommonPrefixLen(SET OF UNSIGNED4 s1, SET OF UNSIGNED4 s2) := BEGINC++
    #option pure
    uint32_t * st1 = (uint32_t *) s1;
    uint32_t * st2 = (uint32_t *) s2;
    uint32_t i;
    uint32_t maxPrefix = lenS1 < lenS2 ? lenS1 / sizeof(uint32_t) : lenS2 / sizeof(uint32_t);
    for (i=0; i < maxPrefix; i++) {
      if (st1[i] != st2[i]) {
        return i;
        }
    }
    return maxPrefix;
    ENDC++;
