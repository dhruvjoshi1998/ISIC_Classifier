import csv
from matplotlib import pyplot
import random

with open('../Data/metadata.csv', mode='r') as f:
    reader = csv.reader(f)
    metadata = [[rows[2], rows[3], rows[4]] for rows in reader if rows[2].isdigit() and rows[3].isdigit() and rows[4] != ""]

random.shuffle(metadata)
val = metadata[int(len(metadata)*.8):]
metadata = metadata[:int(len(metadata)*.8)]

male = [point[:2] for point in metadata if point[2] == 'male']
female = [point[:2] for point in metadata if point[2] == 'female']

age_thresholds = [str(num) for num in range(0,100,5)]

mal = [point for point in metadata if point[0] == '1']

male_ages = []
female_ages = []

male_ages_mal = []
female_ages_mal = []

for age_val in age_thresholds:
	count = 0
	mal_count = 0
	for dp in male:
		if dp[1] == age_val:
			count += 1
			if dp[0] == '1':
				mal_count += 1
	male_ages.append(count)
	male_ages_mal.append(mal_count)

	count = 0
	mal_count = 0
	for dp in female:
		if dp[1] == age_val:
			count += 1
			if dp[0] == '1':
				mal_count += 1
	female_ages.append(count)
	female_ages_mal.append(mal_count)


male_ages_mal = [dp/len(mal) for dp in male_ages_mal]
female_ages_mal = [dp/len(mal) for dp in female_ages_mal]

male_ages = [dp/(len(male)+len(female)) for dp in male_ages]
female_ages = [dp/(len(male)+len(female)) for dp in female_ages]

print(sum(male_ages_mal) + sum(female_ages_mal))
prob_mal = len(mal)/(len(male)+len(female))

probs_mal_given_a_male = [n*prob_mal/d if d != 0 else 0 for n, d in zip(male_ages_mal, male_ages)]
probs_mal_given_a_female = [n*prob_mal/d if d != 0 else 0 for n, d in zip(female_ages_mal, female_ages)]

print(male_ages)
print("----------")
print(male_ages_mal)
print("----------")
print(female_ages)
print("----------")
print(female_ages_mal)
print("----------")

print(probs_mal_given_a_male)
print("----------")
print(probs_mal_given_a_female)

accuracy = 0
count = 0
for dp in metadata:
	age_idx = age_thresholds.index(dp[1])
	if dp[2] == 'male':
		accuracy += int(dp[0])*probs_mal_given_a_male[age_idx] + (1-int(dp[0]))*(1-probs_mal_given_a_male[age_idx])
	elif dp[2] == 'female':
		accuracy += int(dp[0])*probs_mal_given_a_female[age_idx] + (1-int(dp[0]))*(1-probs_mal_given_a_female[age_idx])
	count += 1

accuracy2=0
count2=0
for dp in val:
	age_idx = age_thresholds.index(dp[1])
	thres = 0.5
	if dp[2] == 'male':
		accuracy2 += int(dp[0])*(probs_mal_given_a_male[age_idx]>thres) + (1-int(dp[0]))*(probs_mal_given_a_male[age_idx]<thres)
	elif dp[2] == 'female':
		accuracy2 += int(dp[0])*(probs_mal_given_a_female[age_idx]>thres) + (1-int(dp[0]))*(probs_mal_given_a_female[age_idx]<thres)
	count2 += 1	

print("baseline accuracy is:", 1-(len(mal)/count))
print("calc:", (len(mal)/count)**2 + (1-len(mal)/count)**2)

print("accuracy is:", accuracy/count)
print("accuracy2 is:", accuracy2/count2)

# print(probs_mal_given_a_male[7])
