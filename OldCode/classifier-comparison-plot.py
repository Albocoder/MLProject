import numpy as np
import matplotlib.pyplot as plt; plt.rcdefaults()

accuracies = (78, 41, 52, 41, 45, 47, 45, 50, 40, 48)

objects = ('LinSVM', 'RbfSVM', 'PolySVM', '1-kNN', '3-kNN', '5-kNN', '7-kNN', '9-kNN', 'DTree', 'NBayes')
y_pos = np.arange(len(objects)) - 1


plt.bar(y_pos, accuracies, align='center', alpha=0.4)
plt.xticks(y_pos , objects)
plt.ylabel('Test Accuracies')
plt.title('Accuracies vs Classifiers')

plt.show()


# ind = np.arange(N)  # the x locations for the groups
# width = 0.35       # the width of the bars
#
# fig, ax = plt.subplots()
# rects1 = ax.bar(ind, men_means, width, color='r')
#
# women_means = (93, 38, 93, 89, 88, 84, 83, 80, 54, 24)
# rects2 = ax.bar(ind + width, women_means, width, color='y')
#
# # add some text for labels, title and axes ticks
# ax.set_ylabel('Accuracies')
# ax.set_title('Accuracies of Classifiers')
# ax.set_xticks(ind + width / 2)
# ax.set_xticklabels(('LinSVM', 'RbfSVM', 'PolySVM', '1-kNN', '3-kNN', '5-kNN', '7-kNN', '9-kNN', 'DTree', 'NBayes'))
#
#
# def autolabel(rects):
#     """
#     Attach a text label above each bar displaying its height
#     """
#     for rect in rects:
#         height = int(rect.get_height())
#         ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
#                 '%d' % int(height),
#                 ha='center', va='bottom')
#
# autolabel(rects1)
# autolabel(rects2)
#
# plt.show()
