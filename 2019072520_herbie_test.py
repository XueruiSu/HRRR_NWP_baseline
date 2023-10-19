from herbie import Herbie

# generate a random string
import random
rand_str = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=10))
print(rand_str)
H = Herbie(
    "20190725 20",
    model="hrrr",
    product="prs",
    fxx=0,
    save_dir=f"~/{rand_str}_herbie/"
)
# print(H.xarray())
print(H.inventory(searchString="TMP:2 m above"))
# ds = H.xarray("TMP:2 m above")
ds = H.xarray("TMP:2 m above")
# print(ds.to_array())
print(ds)
numpy_array = ds.to_array().values[0]

print(numpy_array.shape)

print(numpy_array.max(), numpy_array.min(), numpy_array.mean(), numpy_array.std())



