import json


class ParamParse(object):
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.params_struct = json.load(open(self.json_path, "r"))
        self.param_val = {}

    def check_and_parse(self, required_params, required_fields):
        # Check parameters
        # from IPython import embed; embed()
        for param in required_params: 
            assert param in self.params_struct.keys(),f"Missing parameter in param file! Required{param}"
            
        # assert (set(required_params) == set(self.params_struct.keys())), \
        #     "Missing parameters in param file! Required{}".format(required_params)

        for key in list(self.params_struct.keys()):
            # Check keys
            assert (
                list(self.params_struct[key].keys()) == required_fields
            ), "Missing fields in the parameter file for key:{}, required:{}".format(
                key, required_fields
            )
        self._parse_args()

    def get_param_val(self):
        return self.param_val

    def save_struct_to_file(self, file):
        with open(file, "w") as outfile:
            json.dump(self.params_struct, outfile)

    def _parse_args(self):
        for key in list(self.params_struct.keys()):
            if self.params_struct[key]["val"] != "null":
                if self.params_struct[key]["type"] == "int":
                    self.param_val[key] = int(self.params_struct[key]["val"])
                elif self.params_struct[key]["type"] == "tuple-int":
                    self.param_val[key] = tuple(
                        [int(k) for k in self.params_struct[key]["val"]]
                    )
                else:
                    self.param_val[key] = self.params_struct[key]["val"]
            else:
                assert True, "No value entry in {}".format(key)


class GraspParamParse(ParamParse):
    """Param Parser for Grasp Reconstruction"""

    def __init__(self, json_path: str, csv_path: str, subject_id: str, save_path: str):
        super().__init__(json_path)

        self.file_path = json_path
        self.params_struct["path_root_csv"]["val"] = csv_path
        self.params_struct["subject_id"]["val"] = subject_id
        self.params_struct["save_root"]["val"] = save_path

        required_params = [
            "path_root_csv",
            "subject_id",
            "save_root",
            "spv",
            "lam",
            "max_num_iter",
            "max_num_ls",
            "nd",
            "kd",
            "jd",
            "is_gpu",
            "size_nufft_batch",
            "fov_scaling",
        ]
        required_fields = ["section", "helpTip", "val", "type"]
        self.check_and_parse(required_params, required_fields)


class PostProcessParamParse(ParamParse):
    def __init__(self, json_path: str):
        super().__init__(json_path)

        self.file_path = json_path
        required_params = [
            "post_img_size",
            "rate_os",
            "flag_fft_shift",
            "dx",
            "dy",
            "dz",
            "dt",
        ]
        required_fields = ["section", "helpTip", "val", "type"]
        self.check_and_parse(required_params, required_fields)


class RacerGraspParamParse(ParamParse):
    """Param Parser for Grasp Reconstruction"""

    def __init__(self, json_path: str, csv_path: str, subject_id: str, save_path: str):
        super().__init__(json_path)

        self.params_struct["path_root_csv"]["val"] = csv_path
        self.params_struct["subject_id"]["val"] = subject_id
        self.params_struct["save_root"]["val"] = save_path

        required_params = [
            "path_root_csv",
            "subject_id",
            "save_root",
            "spv",
            "lam",
            "max_num_iter",
            "max_num_ls",
            "nd",
            "kd",
            "jd",
            "is_gpu",
            "size_nufft_batch",
            "coil_method",
            "scale_mult",
            "scale_thr",
            "fov_scaling",
        ]
        required_fields = ["section", "helpTip", "val", "type"]
        self.check_and_parse(required_params, required_fields)
