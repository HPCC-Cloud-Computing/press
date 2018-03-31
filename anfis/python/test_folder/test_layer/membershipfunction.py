from skfuzzy import gaussmf, gbellmf, sigmf

x = 3
mean = 10
gamma = 15
a = 3
b = 4
c = 5
# Ham gauss
z1 = gaussmf(x, mean, gamma)

# Ham bell

z2 = gbellmf(x, a, b, c)
print(z2)

# Ham sigmoid 
z3 = sigmf(x, b, c)