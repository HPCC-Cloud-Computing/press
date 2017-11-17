from scale import Scale

data = [8400, 10165, 9482, 8041, 7939, 7608, 8304, 8485, 7278, 7427,
        7556, 7763, 6995, 6593, 6766, 6486, 6120, 6536, 5462, 5906,
        7147, 6074, 6038, 5596, 4529]

system = Scale(10, 1000, 10)
res = system.scale(data, interval=10)

print('Box:', res[0])
print('Cost:', res[1])
print('Request lost:', res[3])
print('Overload time:', res[2])