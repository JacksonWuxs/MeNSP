import bisect
import math


class Simple:
    def __init__(self, variable, missing=""):
        assert isinstance(variable, str)
        assert isinstance(missing, str)
        self.name = variable
        self.miss = missing

    def _render(self, value):
        return value

    def verbalize(self, value):
        if self.miss == value:
            return (self.name, None)
        return (self.name, self._render(value))


class Continue(Simple):
    def __init__(self, variable, missing, splits, groups):
        Simple.__init__(self, variable, missing)
        assert isinstance(groups, list)
        assert isinstance(splits, list)
        assert len(groups) == len(splits) + 1
        self._var = groups
        self._cut = splits

    def _render(self, value):
        return self._var[bisect.bisect_left(self._cut, float(value))]


class Category(Simple):
    def __init__(self, variable, missing, lookup_table):
        assert isinstance(lookup_table, dict)
        Simple.__init__(self, variable, missing)
        self.lookup = lookup_table

    def _render(self, value):
        return self.lookup.get(value)


class Nominal(Simple):
    def __init__(self, variable, missing, lookup_table, split, combine):
        assert isinstance(lookup_table, dict)
        Simple.__init__(self, vairable, missing)
        self.lookup = lookup_table
        self.seg = split
        self.combine = combine

    def _render(self, value):
        return self.combine.join(self.lookup[_] \
                                 for _ in value.split(self.seg)\
                                  if _ in self.lookup)


class Functional(Simple):
    def __init__(self, variable, function, missing=""):
        assert callable(function)
        self._func = function

    def _render(self, value):
        return self._func(value)


class Zipcode(Simple):
    def __init__(self, name, miss):
        Simple.__init__(self, name, miss)
        self._states = {"AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
                        "CA": "California", "CO": "Colorado", "CT": "Connecticut",
                        "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
                        "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
                        "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
                        "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "mississippi", "MO": "Missouri", "MT": "Montana",
                        "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota",
                        "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
                        "PA": "Pennsylvania", "RI": "Rhode Island",
                        "SC": "South Carolina", "SD": "South Dakota",
                        "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
                        "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming"}

    def _render(self, value):
        try:
            return self._states[zipcodes.matching(value)[0]["state"]]
        except Exception:
            return None


class Occupation(Simple):
    def __init__(self, name):
        Simple.__init__(self, name, "none")

    def _render(self, value):
        if value == "other":
            return "freelancer"
        return value

class MovieTitle(Simple):
    def __init__(self, name):
        Simple.__init__(self, name, "N/A")

    def _render(self, value):
        value = value[:-7]
        if "," in value:
            left, right = value.rsplit(",", 1)
            value = right + " " + left
        return value.replace("'", "").replace(".", "")
