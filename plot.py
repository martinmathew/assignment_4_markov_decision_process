import matplotlib.pyplot as plt


values = [-42738.23, -2321.404, -765.135, -770.486, -767.505, -767.68, -771.723, -769.895, -764.604]
discount_factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
fig = plt.figure(figsize=(10, 5))

plt.plot(discount_factors, values)
plt.xlabel('Discount Factor')
plt.ylabel('Average Reward')
plt.title('Average Reward as Function of Discount Factor')
plt.savefig("dfpercentagevsdiscount_factors.png")