import pandas as pd


def sort_by_relationship(atom):
    return int(atom.relationship)


class Atom:
    def __init__(self, relationship, variable1, variable2, relationship_name):
        self.placeholder = 1
        self.relationship = relationship
        self.variable1 = variable1
        self.variable2 = variable2
        self.relationship_name = relationship_name
        self.get_placeholder_variable()

    def get_placeholder_variable(self):
        if self.variable1 == "?g" or self.variable1 == "?h":
            self.placeholder = 0
        else:
            pass

    def __hash__(self):
        return hash(self.id_print())

    def __eq__(self, other):
        if self.variable1 == other.variable1 and self.variable2 == other.variable2 and self.relationship_name == other.relationship_name:
            return True

        return False

    def id_print(self):
        return str(self.relationship) + "(" + self.variable1 + "," + self.variable2 + ")"

    def relationship_print(self):
        return self.relationship_name + "(" + self.variable1 + "," + self.variable2 + ")"

    def neo4j_print(self):
        res = "(" + self.variable1 + ")-[:`" + str(self.relationship) + "`]->(" + self.variable2 + ")"
        res = res.replace("?", "")
        return res

    def get_variables(self):
        return self.variable1, self.variable2


class Rule:
    def __init__(self, head_atom, body_atoms, head_coverage, pca_confidence, functional_variable, beta=1):

        self.head_atom = head_atom
        self.body_atoms = body_atoms
        self.head_coverage = head_coverage
        self.pca_confidence = pca_confidence
        self.selectivity = ((1 + beta * beta) * self.pca_confidence * self.head_coverage) / (
                beta * beta * self.pca_confidence + self.head_coverage)
        self.functional_variable = functional_variable

    def __hash__(self):
        return hash(self.id_print())

    def create_atom_storage_structure(self):
        atom_storage = {}

        for atom in self.body_atoms:
            atom_storage[(atom.variable1, atom.variable2)] = atom.relationship

        atom_storage[(self.head_atom.variable1, self.head_atom.variable2)] = self.head_atom.relationship

        return atom_storage

    def __eq__(self, other):
        if len(self.body_atoms) == len(other.body_atoms):

            atom_flag = [False for i in range(len(self.body_atoms)+1)]

            for i in range(0, len(self.body_atoms)):
                atom_self = self.body_atoms[i]
                for j in range(0, len(other.body_atoms)):
                    atom_other = other.body_atoms[j]
                    if atom_self == atom_other:
                        atom_flag[i] = True
                        break

            atom_flag[-1] = (self.head_atom == other.head_atom)
            for value in atom_flag:
                if not value:
                    return False

            return True
        else:
            return False

    def id_print(self):

        str = ""

        for atom in self.body_atoms:
            str += atom.id_print() + " "

        str += "==>"

        str += self.head_atom.id_print()

        return str

    def relationship_print(self):
        str = ""

        for atom in self.body_atoms:
            str += atom.relationship_print() + " "

        str += "==>"

        str += self.head_atom.relationship_print()

        return str

    def print_metrics(self):
        return "Head Coverage: " + str(self.head_coverage) + ", PCA Confidence: " + str(self.pca_confidence) + \
               ", Selectivity: " + str(self.selectivity)

    def get_variables(self):

        variables = []
        for atom in self.body_atoms:
            v1, v2 = atom.get_variables()
            if v1 not in variables:
                variables.append(v1.replace("?", ""))
            if v2 not in variables:
                variables.append(v2.replace("?", ""))

        return variables


class ParseRule:

    def __init__(self, filename, folder_to_datasets, model_name, dataset_name, relation_delimiter="\t"):
        self.filename = filename
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.rules_by_predicate = {}
        self.rules = []
        self.id_to_relationship = {}
        self.folder_to_datasets = folder_to_datasets
        self.create_id_to_relationship(relation_delimiter)

    def create_id_to_relationship(self, relation_delimiter):
        f_ids = open(f"{self.folder_to_datasets}/{self.dataset_name}/relation2id.txt")
        f_ids.readline()

        for line in f_ids:
            splits = line.strip().split(relation_delimiter)
            self.id_to_relationship[int(splits[1])] = splits[0]

    def parse_rules_from_file(self, beta=1):
        f_rule = open(self.filename, "r")

        num_lines = sum(1 for line in open(self.filename, "r"))

        for ctr in range(0, 15):
            f_rule.readline()

        for ctr in range(15, num_lines - 3):
            line = f_rule.readline()
            splits = line.strip().split("\t")
            functional_variable = splits[-1].replace("?", "")
            body_atoms, head_atom = self.create_atom_from_rule(splits[0])
            head_coverage = float(splits[1])
            pca_confidence = float(splits[3])

            relationship_id = head_atom.relationship

            if relationship_id not in self.rules_by_predicate:
                self.rules_by_predicate[relationship_id] = []

            r = Rule(head_atom, body_atoms, head_coverage, pca_confidence, functional_variable, beta)
            self.rules_by_predicate[relationship_id].append(r)
            self.rules.append(r)

        for pred in self.rules_by_predicate:
            self.rules_by_predicate[pred] = sorted(self.rules_by_predicate[pred], key=lambda x: x.selectivity,
                                                   reverse=True)
        f_rule.close()

    def create_atom_from_rule(self, rule):
        splits = rule.split(" ")
        body_atom_end_index = splits.index('')

        body_atoms = []
        for i in range(0, body_atom_end_index):
            atom_string = splits[i]
            relationship_id = int(atom_string[0:atom_string.index("(")])
            variable1 = atom_string[atom_string.index("(") + 1:atom_string.index(",")]
            variable2 = atom_string[atom_string.index(",") + 1:atom_string.index(")")]
            relationship_name = self.id_to_relationship[relationship_id]
            body_atoms.append(Atom(relationship_id, variable1, variable2, relationship_name))

        atom_string = splits[-1]

        relationship_id = int(atom_string[0:atom_string.index("(")])
        variable1 = atom_string[atom_string.index("(") + 1:atom_string.index(",")]
        variable2 = atom_string[atom_string.index(",") + 1:atom_string.index(")")]
        relationship_name = self.id_to_relationship[relationship_id]
        head_atom = Atom(relationship_id, variable1, variable2, relationship_name)

        return body_atoms, head_atom

    def get_best_rule_by_predicate(self):

        best_rules_by_predicate = {}

        for predicate in self.rules_by_predicate:
            best_rule = self.rules_by_predicate[predicate][0]

            best_rules_by_predicate[predicate] = best_rule

        return best_rules_by_predicate
