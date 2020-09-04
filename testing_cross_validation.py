from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
iris = datasets.load_iris()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(iris.data, iris.target)

sorted(clf.cv_results_.keys())
#['mean_fit_time', 'mean_score_time', 'mean_test_score',...
# 'param_C', 'param_kernel', 'params',...
# 'rank_test_score', 'split0_test_score',...
# 'split2_test_score', ...
# 'std_fit_time', 'std_score_time', 'std_test_score']