import time

from Foofah.foofah_libs import operators as Operations


def create_python_prog(path, input_data=None, output_file=None):
    out_prog = ""
    Operations.PRUNE_1 = False

    for i, n in enumerate(reversed(path)):
        if i > 0:
            params = n.operation[2]
            params_to_apply = []
            out_prog += "t = " + n.operation[0]["name"] + "(t"
            # out_prog += n.operation[0]['name'] + ' '
            for i in range(1, n.operation[0]["num_params"]):
                out_prog += "," + params[i]
                param_to_add = params[i]
                if param_to_add.isnumeric():
                    param_to_add = int(param_to_add)
                params_to_apply += [
                    param_to_add,
                ]

            out_prog += ")\n"
    if output_file:
        fo = open("foo.txt", "w")
        fo.write(out_prog)
    return out_prog
