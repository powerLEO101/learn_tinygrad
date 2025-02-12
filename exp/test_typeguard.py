from typeguard import install_import_hook, typechecked

@typechecked()
def add_one(x:int):
    return x + 1

print(add_one(a:=1.2))
