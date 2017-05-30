IMPORT Std;
EXPORT Bundle := MODULE(Std.BundleBase)
 EXPORT Name := 'LearningTrees';
 EXPORT Description := 'LearningTrees Bundle for Tree-based Machine Learning';
 EXPORT Authors := ['HPCCSystems'];
 EXPORT License := 'http://www.apache.org/licenses/LICENSE-2.0';
 EXPORT Copyright := 'Copyright (C) 2017 HPCC Systems';
 EXPORT DependsOn := ['ML_Core'];
 EXPORT Version := '1.0.0';
 EXPORT PlatformVersion := '6.4.0';
END;