from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np
from scipy import stats
import re
import warnings
import sys
import logging
import copy

logger = logging.getLogger(__name__)
logger.setLevel("INFO")
logging.basicConfig()


class Arff:
    """
    Contains arff file data.
    For discrete attributes, at least one value must be a float in
    order for numpy array functions to work properly. (The load_arff
    function ensures that all values are read as floats.)
    To do: Change backend to use Pandas dataframe
    """

    def __init__(self, arff=None, row_idx=None, col_idx=None, label_count=None, name="Untitled", numeric=True, missing=float("NaN")):
        """
        Args:
            arff (str or Arff object): Path to arff file or another arff file
            row_start (int):
            col_start (int):
            row_count (int):
            col_count (int):
            label_count (int):
        """

        self.data = None
        self.dataset_name = name
        self.attr_names = []
        self.attr_types = []
        self.str_to_enum = []  # list of dictionaries
        self.enum_to_str = []  # list of dictionaries
        self.label_columns = []
        self.MISSING = missing
        self.label_count = label_count
        self.numeric = numeric

        if isinstance(arff, Arff): # Make a copy of arff file
            logger.debug("Creating ARFF from ARFF object")
            if self.dataset_name == "Untitled":
                name = arff.dataset_name+"_subset"
            self._copy_and_slice_arff(arff,row_idx, col_idx, label_count, name)
        elif isinstance(arff, str) or (sys.version_info < (3, 0) and isinstance(arff, unicode)):  # load from path
            logger.debug("Creating ARFF from file path")
            self.load_arff(arff)
            if label_count is None: # if label count is not specified, assume 1
                label_count = 1
                warnings.warn("Label count not specified, using 1")
            self._copy_and_slice_arff(self, row_idx, col_idx, label_count, name)
        elif isinstance(arff, np.ndarray): # convert 2D numpy array to arff
            logger.debug("Creating ARFF from ND_ARRAY")
            self.data = arff
            if label_count is None:
                warnings.warn("Label count not specified, using None")
            self._copy_and_slice_arff(self, row_idx, col_idx, label_count, name)
        else:
            logger.debug("Creating Empty Arff object")
            # Empty arff data structure
            pass

        # Initialize vacuous if data defined
        if not self.data is None:
            columns = self.data.shape[1]
            self.attr_names = [x for x in range(columns)]         if not self.attr_names else self.attr_names
            self.attr_types = ["Unknown" for x in range(columns)] if not self.attr_types else self.attr_types
            self.str_to_enum = [{} for x in range(columns)]       if not self.str_to_enum else self.str_to_enum
            self.enum_to_str = [{} for x in range(columns)]       if not self.enum_to_str else self.enum_to_str
            self.label_columns = []

    def set_size(self, rows, cols):
        """Resize this matrix (and set all attributes to be continuous)"""
        self.data = np.zeros((rows, cols))
        self.attr_names = [""] * cols
        self.str_to_enum = []
        self.enum_to_str = []

    def load_arff(self, filename):
        """Load matrix from an ARFF file"""
        self.data = None
        self.attr_names = []
        self.str_to_enum = []
        self.enum_to_str = []
        reading_data = False

        rows = []  # we read data into array of rows, then convert into array of columns

        with open(filename) as f:
            for line in f.readlines():
                line = line.strip()  # why was this rstrip()?
                if len(line) > 0 and line[0] != '%':
                    if not reading_data:
                        if line.lower().startswith("@relation"):
                            self.dataset_name = line[9:].strip()
                        elif line.lower().startswith("@attribute"):
                            attr_def = line[10:].strip()
                            if attr_def[0] == "'":
                                attr_def = attr_def[1:]
                                attr_name = attr_def[:attr_def.index("'")]
                                attr_def = attr_def[attr_def.index("'") + 1:].strip()
                            else:
                                search = re.search(r'(\w*)\s*(.*)', attr_def)
                                attr_name = search.group(1)
                                attr_def = search.group(2)
                                # Remove white space from atribute values
                                attr_def = "".join(attr_def.split())

                            self.attr_names += [attr_name]

                            str_to_enum = {}
                            enum_to_str = {}
                            if attr_def.lower() in ["real", "continuous"]:
                                self.attr_types.append("continuous")
                            elif attr_def.lower() == "integer":
                                self.attr_types.append("ordinal")
                            else:
                                # attribute is discrete
                                assert attr_def[0] == '{' and attr_def[-1] == '}'
                                attr_def = attr_def[1:-1]
                                attr_vals = attr_def.split(",")
                                val_idx = 0
                                for val in attr_vals:
                                    val = val.strip()
                                    enum_to_str[val_idx] = val
                                    str_to_enum[val] = val_idx
                                    val_idx += 1
                                self.attr_types.append("nominal")
                            self.enum_to_str.append(enum_to_str)
                            self.str_to_enum.append(str_to_enum)

                        elif line.lower().startswith("@data"):
                            reading_data = True

                    else:
                        # reading data
                        val_idx = 0
                        # print("{}".format(line))
                        vals = line.split(",")
                        if self.numeric:
                            row = np.zeros(len(vals))
                        else:
                            row = np.empty(len(vals), dtype=object)

                        for i,val in enumerate(vals):
                            val = val.strip()
                            if not val:
                                raise Exception("Missing data element in row with data '{}'".format(line))
                            else:
                                if self.numeric: # record indices for nominal variables
                                    row[val_idx] = float(
                                        self.MISSING if val == "?" else self.str_to_enum[val_idx].get(val, val))
                                else: # record actual values
                                    row[val_idx] = self.MISSING if val == "?" else val


                                # Capture missings in str_to_enum
                                # if val == "?" and self.str_to_enum[i] and not "?" in self.str_to_enum:
                                #
                                #     num = max(self.str_to_enum[i].values())
                                #     self.str_to_enum[i]["?"] = num
                                #     self.enum_to_str[i][num] = "?"

                            val_idx += 1
                        rows += [row]
        self.data = np.array(rows)


    @property
    def instance_count(self):
        """Get the number of rows in the matrix"""
        return self.data.shape[0]

    @property
    def features_count(self):
        """Get the number of columns (or attributes) in the matrix"""
        return self.data.shape[1] - self.label_count

    def create_subset_arff(self, row_idx=None, col_idx=None, label_count=None):
        """ This returns a new arff file with specified slices; both objects reference same underlying data
        Args:
            row_idx (slice() or list): A slice or list of row indices
            col_idx (slice() or list):  A slice or list of col indices
            label_count (int): The number of columns to be considered as "labels"
        Returns:
        """
        new_arff = Arff(arff=self, row_idx=row_idx, col_idx=col_idx, label_count=label_count) # create a copy
        return new_arff

    def _copy_and_slice_arff(self, arff=None, row_idx=None, col_idx=None, label_count=None, dataset_name="Untitled"):
        """ This copies an external arff to the current arff object, slicing as specified
        Args:
            row_idx (slice() or list): A slice or list of row indices
            col_idx (slice() or list):  A slice or list of col indices
            label_count (int): The number of columns to be considered as "labels"
        Returns:
        """
        def slicer(_list, idx):
            """ If a list is specified as a slice, loop through. Idx should be a list, int, or slice.
                Returns:
                    A list!!
            """
            try:
                if isinstance(col_idx, list):
                    return [_list[i] for i in col_idx]
                elif isinstance(col_idx, int):
                    return [_list[idx]]
                elif isinstance(col_idx, slice):
                    return _list[idx]
                else:
                    raise Exception("Unexpected index type")
            except:
                warnings.warn("Could not slice {} element of Arff object, returning None".format(_list))
                return None

        if self.is_iterable(row_idx) and self.is_iterable(col_idx):
            warnings.warn("User is attempting to slice both axes using lists. This will result in a 1D array, " \
                          "is not supported by the toolkit, and may not be what the user intended.")

        # Fix row indices
        if row_idx is None:
            row_idx=slice(0,None)
        elif isinstance(row_idx, int):
            row_idx = slice(row_idx,row_idx+1) # make it a list, to preserve dimension
        if col_idx is None:
            col_idx=slice(0,None)
        elif isinstance(col_idx, int):
            col_idx = slice(col_idx,col_idx+1)

        # If reference has label count, but current one doesn't, infer it
        column_count = arff.shape[1]
        if label_count is None and arff.label_count:
            label_list = [1 if i in range(column_count-arff.label_count, column_count) else 0 for i in range(column_count) ]
            self.label_count = sum(slicer(label_list, col_idx))
        else:
            self.label_count = label_count

        ## Update main numpy array
        self.data = arff.data[row_idx, col_idx]
        if len(self.shape) < 2:
            warnings.warn("Unexpected array dimension (should be 2, not {})".format(len(self.shape)))

        ## Update all other features
        self.dataset_name = dataset_name
        self.attr_names = slicer(arff.attr_names,col_idx)
        self.attr_types = slicer(arff.attr_types,col_idx)
        self.str_to_enum = slicer(arff.str_to_enum,col_idx)
        self.enum_to_str = slicer(arff.enum_to_str,col_idx)

    def get_features(self, row_idx=None):
        """ Return features as 2D array
        Args:
            _type: Optionally specify 'nominal' or 'continuous' to return appropriate subset of features
        Returns:
        """
        if row_idx is None:
            row_idx = slice(0,None)
        end_idx = None if self.label_count == 0 else -self.label_count # return all if no labels
        return self.create_subset_arff(row_idx=row_idx, col_idx=slice(0,end_idx), label_count=0)

    def get_labels(self, row_idx=None):
        if row_idx is None:
            row_idx = slice(0,None)

        start_idx = self.shape[1] if -self.label_count == 0 else -self.label_count # return nothing if no labels
        new_arff = self.create_subset_arff(row_idx=row_idx, col_idx=slice(start_idx, None), label_count=self.label_count)
        return new_arff

    def attr_name(self, col):
        """Get the name of the specified attribute"""
        return self.attr_names[col]

    def set_attr_name(self, col, name):
        """Set the name of the specified attribute"""
        self.attr_names[col] = name

    def get_attr_names(self):
        return self.attr_names

    def attr_value(self, attr, val):
        """
        Get the name of the specified value (attr is a column index)
        :param attr: index of the column
        :param val: index of the value in the column attribute list
        :return:
        """
        return self.enum_to_str[attr][val]

    def unique_value_count(self, col=0):
        """
        Get the number of values associated with the specified attribute (or columnn)
        0=continuous, 2=binary, 3=trinary, etc.
        """
        values = len(self.enum_to_str[col]) if self.enum_to_str else 0
        return values

    def is_nominal(self, col=0):
        nominal =self.unique_value_count(col) > 0
        return nominal

    def get_arff_as_string(self):
        """ Print arff class as arff-style string
            Returns:
                string
        """
        out_string = ""
        out_string += "@RELATION {}".format(self.dataset_name) + "\n"
        for i in range(len(self.attr_names)):
            out_string += "@ATTRIBUTE {}".format(self.attr_names[i])
            if self.is_nominal(i):
                out_string += (" {{{}}}".format(", ".join(self.enum_to_str[i].values()))) + "\n"
            else:
                out_string += (" CONTINUOUS") + "\n"

        out_string += ("@DATA") + "\n"

        ## i idx
        for i in range(self.shape[0]):
            r = self.data[i]
            values = []

            # j idx
            for j in range(len(r)):
                if not self.is_nominal(j):
                    if not self.is_missing(r[j]):
                        values.append(str(r[j]))
                    else:
                        values.append("?")
                else:
                    try:
                        if self.numeric:
                            values.append(self.enum_to_str[j][r[j]])
                        else:
                            values.append(r[j])
                    except(Exception) as e:
                        #print(out_string,values)
                        if self.is_missing(r[j]):
                            values.append("?")
                        else:
                            raise e

            # values = list(map(lambda j: str(r[j]) if self.value_count(j) == 0 else self.enum_to_str[j][r[j]],
            #                   range(len(r))))
            out_string += ("{}".format(", ".join(values))) + "\n"

        return out_string

    def __str__(self):
        return self.get_arff_as_string()

    def print(self):
        print(self)

    def nd_array(self, obj):
        """ Convert an arff, list, or numpy array to numpy array
        Args:
            obj (array-like): An object to be converted
        Returns
            numpy array
        """

        if isinstance(obj, Arff):
            return obj.data
        elif isinstance(obj, list):
            return np.ndarray(obj)
        elif isinstance(obj, np.ndarray):
            return obj
        else:
            raise Exception("Unrecognized data type")

    def get_nominal_idx(self):
        nominal_idx = [i for i,feature_type in enumerate(self.attr_types) if feature_type=="nominal"]
        return nominal_idx if nominal_idx else None

    def reshape(self, tup):
        if self.is_iterable(tup):
            return self.data.reshape(*tup)
        return self.data.reshape(tup)

    def __getitem__(self, index):
        """ Trivial wrapper for the 2D Numpy array data
        Args:
            index: Index, slice, etc. whatever you would use for Numpy array
        Returns:
            array-like object
        """

        ## This will slice ARFF and return smaller arffs; it's considerably slower than numpy slicing
        # if not self.is_iterable(index):
        #     index = [index, slice(0,None)]
        # x = self.create_subset_arff(index[0], index[1])
        # return x
        return self.data[index]

    def __setitem__(self, key, value):
        self.data[key] = value

    def copy(self):
        return copy.deepcopy(self)

    def is_iterable(self, obj):
        try:
            iter(obj)
        except TypeError as te:
            return False
        return True

    def __iter__(self):
        """
        Trivial wrapper for looping Numpy 2D array
        """
        for i in self.data:
            yield i

    @property
    def T(self):
        return self.data.T

    def get_dataframe(self):
        import pandas as pd
        df = pd.DataFrame(data=self.data, columns=self.attr_names)
        return df

    @property
    def shape(self):
        return self.data.shape
    # __iter__() and __getitem__()

    def is_missing(self, value):
        if self.MISSING in [np.inf, "?"]:
            return value == self.MISSING
        elif np.isnan(self.MISSING):
            return np.isnan(value)
