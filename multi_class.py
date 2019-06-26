import pandas as pd
import numpy as np
import csv
import random
import math
import time
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report, confusion_matrix 

def read_data(location):
	df = pd.read_csv(location + '.csv' , header = None)
	row = df.shape[0]
	col = df.shape[1]
	print("Row " + str(row) + "  " + "Col " +str(col))
	df.head(10)
	X = df.iloc[0:,:-1].values
	y = df.iloc[0:,col-1].values
	return X,y

def dist(s1 , s2):
    d = len(s1)
    sum = 0.0 
    for i in range(d):
        sum += np.square(s1[i] - s2[i])
    sum = np.sqrt(sum)
    return sum

def neighbor_set(v, X, K):
	samples = X.shape[0]
	d = []
	pos = []
	n = []
	for i in range(samples):
		if dist(X[i],v) != 0:
			d.append(dist(X[i], v))
			pos.append(i)
	d,pos = zip(*sorted(zip(d,pos)))
	for i in range(K):
		n.append(pos[i])
	return n

def victim_samples(frac, samples):
	f = open("victim.txt","w")
	# victims = []
	# v = math.floor(frac*samples)
	# victims = random.sample(range(0,samples-1),v)
	# for i in (victims):
	# 	f.write(str(i) + "\n")
	victims = []
	for i in range(samples):
		victims.append(i)
	return victims

def count_sample(classes, y):
	sample_count = [0 for x in range(classes)]
	for i in y:
		sample_count[i-1] += 1
	return sample_count

def score_matrix_generation(victims, score_matrix, classes, X, y, K):
	q = 0
	p = 0
	samples = X.shape[0]
	for i in victims:
		label = y[i]
		neighbor = neighbor_set(X[i], X, K)
		label_count = [0 for x in range(classes)]
		for n in neighbor:
			label_count[y[n]-1] += 1
		m = max(label_count)
		predicted_label = label_count.index(m)
		if predicted_label != label - 1:
			q =q + 1
			for n in neighbor:
				if y[n] - 1 == label - 1:
					score_matrix[n][label-1][1] += 1 +  neighbor.index(n) + K - label_count[y[n]-1]
				else:
					score_matrix[n][label-1][0] += K - neighbor.index(n) + label_count[y[n]-1]
		if predicted_label == label - 1:
			p = p + 1
			for n in neighbor:
				if y[n] - 1 == label - 1:
					score_matrix[n][label-1][1] += 1 + neighbor.index(n) + K - label_count[y[n]-1]
				else:
					score_matrix[n][label-1][0] += K - neighbor.index(n) + label_count[y[n]-1]
	print("Misclassify ", q)
	print("Classify ", p)
	return score_matrix

def error_bound(score_matrix, samples, classes, K, y):
	misclassify_prob = []
	sample_count = count_sample(classes, y)
	for i in range(samples):
		label = y[i]
		total_prob = 0.0
		for j in range(classes):
			if j != label - 1:
				if (score_matrix[i][j][0] + score_matrix[i][j][1]) != 0:
					class_wise_prob = (score_matrix[i][j][0] / (score_matrix[i][j][0] + score_matrix[i][j][1]))
					total_prob += ((class_wise_prob)*(sample_count[j] / samples))
		misclassify_prob.append(total_prob)
	new_labels = y
	f = open("prob.txt","w")
	for i in range(samples):
		f.write(str(i+1) + "  " + str(misclassify_prob[i]) +  "   " + str(new_labels[i]) +  "\n")
	f.close()
	misclassify_prob, new_labels = zip(*sorted(zip(misclassify_prob, new_labels)))
	f = open("prob_sorted.txt","w")
	for i in range(samples):
		f.write(str(i+1) + "  " + str(misclassify_prob[i]) +  "   " + str(new_labels[i]) +  "\n")
	f.close()
	prob_count = [0 for x in range(classes)]
	print("Calculating theoretical error bound......")
	for c in range(classes):
		max_prob = 0.0
		for t in range(math.floor(K/classes) + 1, K+1):
			pos = 0
			k = 0
			temp = 0.0
			l_count = [0 for x in range(classes)]
			for i in range(samples):
				if new_labels[samples-i-1] == c + 1:
					temp += (misclassify_prob[samples-i-1]*(K-k))
					pos = pos + (K-k)
					k = k + 1
				if k == t:
					break
			if k < K:
				for i in range(samples):
					if new_labels[samples-i-1] != c + 1:
						l_count[c] += 1
						temp += (misclassify_prob[samples-i-1]*(K-k))
						pos = pos + (K-k)
						k = k + 1
					if k == K:
						break
			if pos != 0:
				temp = temp / pos
				if temp > max_prob and t >= max(l_count):
					max_prob = temp
		prob_count[c] = max_prob
	max_bound = 0.0
	for i in range(classes):
		bound = sum(prob_count) - prob_count[i]
		if bound > max_bound:
			max_bound = bound
	theoretical_bound = (max_bound/(classes-1))
	return round(theoretical_bound,4)*100


def write_score_matrix(score_matrix):
	fileptr = open("Score_Matrix.txt","w")
	for i in range(samples):
		fileptr.write(str(i+1) + "     ")
		for j in range(classes):
			fileptr.write(str(score_matrix[i][j]) + "   ") 
		fileptr.write(str(y[i]))
		fileptr.write("\n")

def package_based_knn(K, X, y, val_size):
    count  = 0
    experimental_bound = 0.0
    print("Calculating experimental bound......\n")
    c = []
    ex = []
    while count < 100:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=val_size)
        classifier = KNeighborsClassifier(n_neighbors=K)  
        classifier.fit(X_train, y_train)  
        y_pred = classifier.predict(X_test)  
        error = np.mean(y_test != y_pred)
        count += 1
        c.append(count)
        ex.append(round(error,4)*100)
        experimental_bound += error
    exp_bound = experimental_bound/count
    return round(exp_bound,4)*100,c,ex

# ---------------------------------------------------------------------------------------
# Input Parameters
classes = 3
frac= 0.99
location = r'C:\Users\user\sublime_files\data\tae\tae'
# location = r'C:\Users\user\sublime_files\random_data\rand_data1'
# ---------------------------------------------------------------------------------------

X,y = read_data(location)
samples = X.shape[0]
features = X.shape[1]
flag = 0
for i in range(samples):
	if y[i] == 0:
		flag = 1
		break
if flag == 1:
	for i in range(samples):
		y[i] += 1

print("No of samples  " + str(samples))
print("No of features  " + str(features))
print("No of classes  " + str(classes))
# ----------------------------------------------------------------------------------------

choice = input("\nEnter 1 to run on single K value\n" + "Enter 2 to run on multiple K value\n")
if choice == '1':
	K = input("Enter value of K ")
	print("value of K  " + str(K))
	K = int(K)
	c = []
	ex = []
	th = []
	score_matrix =[[[0,0] for x in range(classes)] for y in range(samples)]
	if samples > 1000:
		print("Huge number of samples......\n")
		frac = float(input("Enter fraction of samples to run...\n"))
		iterations = int(input("Enter number of iterations...\n"))
		iter = 0
		max_th_bound = 0.0
		while iter < iterations:
			print("Iteration  " + str(iter+1))
			victims = victim_samples(frac, samples)
			print("No of victims " + str(len(victims)))
			score_matrix = score_matrix_generation(victims, score_matrix, classes, X, y, K)
			write_score_matrix(score_matrix)
			th_bound = error_bound(score_matrix, samples, classes, K, y)
			iter += 1
			print("Error bound  ", th_bound)
			if th_bound > max_th_bound:
				max_th_bound = th_bound
			print("--------------------------------------------------------")
		theoretical_bound = max_th_bound
		experimental_bound, c, ex = package_based_knn(K, X, y, 0.10)
	else:
		victims = victim_samples(frac, samples)
		print("Victims  " + str(len(victims)))
		score_matrix = score_matrix_generation(victims, score_matrix, classes, X, y, K)
		write_score_matrix(score_matrix)
		theoretical_bound = error_bound(score_matrix, samples, classes, K, y)
		experimental_bound, c, ex = package_based_knn(K, X, y, 0.10)
	th = [theoretical_bound for x in range(len(c))]
	plt.plot(c, th, color = 'blue')
	plt.plot(c, ex, color = 'red')
	plt.ylim(0,100)
	print("Theoretical Bound  " + str(theoretical_bound))
	print("Experimental Bound  " + str(experimental_bound))
	plt.savefig(r"C:\Users\user\sublime_files\results\multi_class_fixed_K.png")
	plt.show()

else:
	th = []
	ex = []
	x = []
	c = []
	exp = []
	stop = math.floor(np.sqrt(samples)) + 1
	for i in range(1,stop,1):
		print("Value of K  " + str(i))
		score_matrix =[[[0,0] for x in range(classes)] for y in range(samples)]
		victims = victim_samples(frac, samples)
		print("Victims  " + str(len(victims)))
		score_matrix = score_matrix_generation(victims, score_matrix, classes, X, y, i)
		write_score_matrix(score_matrix)
		theoretical_bound = error_bound(score_matrix, samples, classes, i, y)
		experimental_bound, c, exp = package_based_knn(i, X, y, 0.10)
		th.append(theoretical_bound)
		ex.append(experimental_bound)
		x.append(i)
		print("Theoretical Bound  " + str(theoretical_bound))
		print("Experimental Bound  " + str(experimental_bound))
		print("----------------------------------------------------------")
	print(np.corrcoef(th,ex))
	print("Avarage Theoretical Bound ", np.mean(th))
	print("Avarage Experimental Bound ", np.mean(ex))
	plt.plot(x, th, color = "blue")
	plt.plot(x, ex, color = "red")
	plt.xlabel("values of K")
	plt.ylabel("error bounds")
	plt.ylim(0,100)
	plt.savefig(r"C:\Users\user\sublime_files\results\multi_class.png")
	plt.show()

# -----------------------------------------------------------------------------------------------------
