import pandas

mydataset = {
    'cars' : ["Hyundai", "BMW", "Mercedes"],
    'yearOfManifacturing' : [2000, 2003, 1990]
}

myvar = pandas.DataFrame(mydataset)

print(myvar)