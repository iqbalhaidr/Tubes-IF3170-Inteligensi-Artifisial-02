import numpy as np
import pickle

class TreeNode:
    def __init__(self, is_leaf, classification = None, attribute = None, threshold = None, branches = None, len_instances = 0):
        self.is_leaf = is_leaf
        self.classification = classification
        self.attribute = attribute
        self.threshold = threshold
        self.branches = branches if branches is not None else {}
        self.len_instances = len_instances

class Rule:
    def __init__(self, conditions, classification, accuracy=0.0, y_val_covered=0):
        self.conditions = conditions # list (attribute, operator, value)
        self.classification = classification
        self.accuracy = accuracy
        self.y_val_covered = y_val_covered #instance dari dataset validasi

    # cek instance validasi masuk ke dalam rule
    def is_matched(self, instance):
        for attribute, operator, value in self.conditions:
            # atribut numerikal -> value merupakan threshold
            if operator == '<=' and instance[attribute] > value:
                return False
            elif operator == '>' and instance[attribute] <= value:
                return False
            elif operator == '==' and instance[attribute] != value: # atribut kategorikal
                return False
        return True

    # def formatted_string(self):
    #     cond_str = " AND ".join([f"X[{a}] {op} {v}" for a, op, v in self.conditions])
    #     return f"IF {cond_str} THEN {self.classification} (acc={self.accuracy:.3f}, cov={self.coverage})"


class DecisionTreeLearning:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, pruning='none'):
        self.tree = None
        self.classes = None
        self.default_class = None
        self.pruning = pruning
        self.n_features = 0
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.rules = [] # Buat rule post pruning


    # Hitung entropy dari examples
    def calculate_entropy(self, examples):
        y = examples[:, -1]
        if len(y) == 0:
            return 0

        # Ambil jumlah class dari examples
        unique_classes, counts = np.unique(y, return_counts=True)

        entropy = 0
        for count in counts:
            probability = count / len(y)
            if probability > 0:
                entropy -= probability * np.log2(probability)

        return entropy

    # hitung information gain pada atribut tertentu
    def calculate_information_gain(self, examples, attribute, threshold):
        X_a = examples[:, attribute]
        parent_entropy = self.calculate_entropy(examples)

        len_S = len(examples)
        weighted_entropy = 0

        # Perhtitungan dari split threshold untuk atribut numerik
        if threshold is not None:
            left_indices = X_a <= threshold
            right_indices = X_a > threshold
            left_subset, right_subset  = examples[left_indices], examples[right_indices]

            weight_left = len(left_subset) / len_S
            weight_right = len(right_subset) / len_S

            weighted_entropy += (weight_left * self.calculate_entropy(left_subset) +
                                    weight_right * self.calculate_entropy(right_subset))
        else:
            # Ambil nilai unik pada atribut
            unique_values = np.unique(X_a)
            for value in unique_values:
                # ambil index untuk nilai atribut yg cocok sama value
                indices = np.where(X_a == value)[0]
                subset = examples[indices]

                weight = len(subset) / len_S
                weighted_entropy += weight * self.calculate_entropy(subset)

        information_gain = parent_entropy - weighted_entropy
        return information_gain

    # Hitung split information dari atribut tertentu
    def calculate_split_information(self, examples, attribute, threshold):
        X_a = examples[:, attribute]
        len_S = len(examples)
        split_info = 0

        # Pecah atribut numerik jadi diskrit berdasarkan threshold
        if threshold is not None:
            len_S_left = np.sum(X_a <= threshold)
            len_S_right = np.sum(X_a > threshold)

            proportion_left = len_S_left / len_S
            proportion_right = len_S_right / len_S

            if proportion_left > 0:
                split_info -= proportion_left * np.log2(proportion_left)
            if proportion_right > 0:
                split_info -= proportion_right * np.log2(proportion_right)

        else:
            unique_values = np.unique(X_a)
            for value in unique_values:
                len_Si = np.sum(X_a == value)
                proportion = len_Si / len_S
                if proportion > 0:
                    split_info -= proportion * np.log2(proportion)

        return split_info

    # Hitung gain ratio buat atribut tertentu
    def calculate_gain_ratio(self, examples, attribute, threshold = None):

        info_gain = self.calculate_information_gain(examples, attribute, threshold)
        split_info = self.calculate_split_information(examples, attribute, threshold)

        if split_info == 0:
            return 0

        gain_ratio = info_gain / split_info

        return gain_ratio

    # Cari threshold terbaik dari beberapa kandidat threshold (numerik)
    def find_best_threshold(self, examples, attribute):
        # index atribut yang udah terurut buat valuenya
        X_a = examples[:, attribute]
        y = examples[:, -1]

        sorted_indices = np.argsort(X_a)
        X_sorted = X_a[sorted_indices]
        y_sorted = y[sorted_indices]

        threshold_candidates = []
        for i in range(len(X_sorted) - 1):
            # Ambil threshold dari midway dua nilai yang berbeda kelasnya
            if y_sorted[i] != y_sorted[i + 1]:
                threshold = (X_sorted[i] + X_sorted[i + 1]) / 2
                threshold_candidates.append(threshold)

        if (len(threshold_candidates) == 0):
            return None, 0

        best_threshold = None
        best_gain_ratio = -9999999

        # Cari threshold terbaik berdasarkan gain ratio (C4.5)
        for threshold in threshold_candidates:
           gain_ratio = self.calculate_gain_ratio(examples, attribute,  threshold)
           if gain_ratio > best_gain_ratio:
               best_gain_ratio = gain_ratio
               best_threshold = threshold

        return best_threshold, best_gain_ratio

    # Cari kelas paling banyak dari examples
    def plurality_value(self, examples):
        y = examples[:, -1]

        unique_classes, counts = np.unique(y, return_counts=True)
        most_common_class = unique_classes[np.argmax(counts)]
        return most_common_class

    def importance(self, attribute, examples):
        X = examples[:, :-1]
        y = examples[:, -1]

        X_a = X[:, attribute]

        # Cek apakah atribut numerik atau kategorik
        if np.issubdtype(X_a.dtype, np.number):
            threshold, gain_ratio = self.find_best_threshold(examples, attribute)
        else:
            threshold = None
            gain_ratio = self.calculate_gain_ratio(examples, attribute, threshold=None)

        return gain_ratio, threshold

    def argmax_importance(self, examples, attributes):
        best_attribute = None
        best_threshold = None
        best_gain_ratio = -9999999

        for attribute in attributes:
            gain_ratio, threshold = self.importance(attribute, examples)

            if gain_ratio > best_gain_ratio:
                best_gain_ratio = gain_ratio
                best_attribute = attribute
                best_threshold = threshold

        return best_attribute, best_threshold

    def build_decision_tree(self, examples, attributes, parent_examples, depth=0):
        # Base case
        # Examples kosong
        if len(examples) == 0:
            most_common_class = self.plurality_value(parent_examples)
            return TreeNode(is_leaf=True, classification=most_common_class)

        # Kondisi stop untuk max depth atau min_samples_split
        if (self.max_depth is not None and depth >= self.max_depth) or (len(examples) < self.min_samples_split):
            most_common_class = self.plurality_value(examples)
            return TreeNode(is_leaf=True, classification=most_common_class, len_instances=len(examples))


        # Semua examples punya klasifikasi sama
        if len(np.unique(examples[:, -1])) == 1:
            exs_classification = examples[0, -1]
            return TreeNode(is_leaf=True, classification=exs_classification, len_instances = len(examples))

        # Attribute empty
        if len(attributes) == 0:
            most_allcommon_class = self.plurality_value(examples)
            return TreeNode(is_leaf=True, classification=most_allcommon_class, len_instances = len(examples))

        # Recursive case
        best_attribute, best_threshold = self.argmax_importance(examples, attributes)
        remaining_attributes = attributes - {best_attribute}

        tree = TreeNode(is_leaf=False, attribute=best_attribute, threshold=best_threshold, len_instances = len(examples))

        # Buat dua branches (<=, > dari threshold) untuk atribut numerik
        if best_threshold is not None:
            left_indices = examples[:, best_attribute] <= best_threshold
            right_indices = examples[:, best_attribute] > best_threshold
            left_subset_examples = examples[left_indices]
            right_subset_examples = examples[right_indices]

            if len(left_subset_examples) < self.min_samples_leaf or len(right_subset_examples) < self.min_samples_leaf:
                most_common_class = self.plurality_value(examples)
                return TreeNode(is_leaf=True, classification=most_common_class, len_instances=len(examples))

            left_subtree = self.build_decision_tree(left_subset_examples, remaining_attributes, parent_examples = examples, depth=depth + 1)
            right_subtree = self.build_decision_tree(right_subset_examples, remaining_attributes, parent_examples = examples, depth=depth + 1)

            tree.branches['<='] = left_subtree
            tree.branches['>'] = right_subtree

        # Buat branch dari tree sebanyak nilai unik atribut kategorik
        else:
            X = examples[:, :-1]
            unique_values = np.unique(X[:, best_attribute])

            # bikin value dan subset dalam dictionary
            value_subsets = {}
            is_split_valid = True

            for value in unique_values:
                indices = np.where(X[:, best_attribute] == value)[0]
                subset_examples = examples[indices]

                if len(subset_examples) < self.min_samples_leaf:
                    is_split_valid = False
                    break

                # Simpan value dan subset_examples sebagai tuple
                value_subsets[value] = subset_examples

            # Jika split ga valid, jadikan leaf
            if not is_split_valid:
                most_common_class = self.plurality_value(examples)
                return TreeNode(is_leaf=True, classification=most_common_class, len_instances=len(examples))

            for value, subset_examples in value_subsets.items():
                subtree = self.build_decision_tree(subset_examples, remaining_attributes, parent_examples=examples, depth=depth + 1)
                tree.branches[value] = subtree

        return tree

    # convert tree jadi rules buat rule pruning
    def tree_to_rules(self, node, conditions=[]):
        # base case
        if node.is_leaf:
            rule = Rule(conditions=conditions.copy(), classification=node.classification)
            self.rules.append(rule)
            return

        # rekursif untuk atribut numerik
        if node.threshold is not None:
            if '<=' in node.branches:
                left_conditions = conditions + [(node.attribute, '<=', node.threshold)]
                self.tree_to_rules(node.branches['<='], left_conditions)
            if '>' in node.branches:
                right_conditions = conditions + [(node.attribute, '>', node.threshold)]
                self.tree_to_rules(node.branches['>'], right_conditions)

        else: # buat kategorikal
            for value, child in node.branches.items():
                new_conditions = conditions + [(node.attribute, '==', value)]
                self.tree_to_rules(child, new_conditions)


    def calculate_rule_accuracy(self, rule, examples):
        X = examples[:, :-1]
        y = examples[:, -1]

        y_covered = []
        for i in range(len(examples)):
            if rule.is_matched(X[i]):
                y_covered.append(y[i])

        if len(y_covered) == 0:
            return 0.0, 0

        # Hitung akurasi dataset validasi
        n_true = sum([1 for c in y_covered if c == rule.classification])
        accuracy = n_true / len(y_covered)
        return accuracy, len(y_covered)

    # Helper untuk proses pruning rule dengan remove preconditions nya
    def prune_rule(self, rule, val_examples):

        if len(rule.conditions) == 0:
            return rule

        best_rule = rule
        best_accuracy, best_y_covered = self.calculate_rule_accuracy(rule, val_examples)
        best_rule.accuracy = best_accuracy
        best_rule.y_val_covered = best_y_covered

        accuracy_improved = True
        while accuracy_improved and len(best_rule.conditions) > 0:
            accuracy_improved = False

            # Buat rule baru dengan menghapus condition secara greedy
            for i in range(len(best_rule.conditions)):
                # hapus condition i
                new_conditions = best_rule.conditions[:i] + best_rule.conditions[i+1:]
                new_rule = Rule(conditions=new_conditions, classification=best_rule.classification)

                accuracy, y_covered = self.calculate_rule_accuracy(new_rule, val_examples)

                if accuracy >= best_accuracy:
                    best_rule = new_rule
                    best_accuracy = accuracy
                    best_y_covered = y_covered
                    accuracy_improved = True  # tetap di dalam loop outer
                    break

        best_rule.accuracy = best_accuracy
        best_rule.y_val_covered = best_y_covered
        return best_rule

    # post pruning dengan rule
    def post_pruning_by_rule(self, val_examples):

        self.rules = []
        self.tree_to_rules(self.tree)

        pruned_rules = []
        for rule in self.rules:
            pruned_rule = self.prune_rule(rule, val_examples)
            pruned_rules.append(pruned_rule)

        # sort rule by accuracy
        accuracies = np.array([rule.accuracy for rule in pruned_rules])
        sorted_indices = np.argsort(accuracies)[::-1]
        pruned_rules = [pruned_rules[i] for i in sorted_indices]

        self.rules = pruned_rules


    # post pruning dengan reduced error
    def reduced_error_pruning(self, node, val_examples):
        if node.is_leaf:
            return node

        # Buat atribut numerik
        if node.threshold is not None:
            left_indices = val_examples[:, node.attribute] <= node.threshold
            right_indices = val_examples[:, node.attribute] > node.threshold

            # rekursif untuk <= threshold
            if '<=' in node.branches and np.sum(left_indices) > 0:
                node.branches['<='] = self.reduced_error_pruning(
                    node.branches['<='], val_examples[left_indices]
                )
            # rekursif untuk > threshold
            if '>' in node.branches and np.sum(right_indices) > 0:
                node.branches['>'] = self.reduced_error_pruning(
                    node.branches['>'], val_examples[right_indices]
                )
        else: # buat atribut kategorikal
            X = val_examples[:, :-1]
            for value, child in list(node.branches.items()):
                indices = np.where(X[:, node.attribute] == value)[0]
                # lakukan rekursif kalau example nya ga kosong
                if len(indices) > 0:
                    node.branches[value] = self.reduced_error_pruning(
                        child,
                        val_examples[indices]
                    )

        X_val = val_examples[:, :-1]
        y_val = val_examples[:, -1]

        # Hitung subtree accuracy dari kecocokan hasil prediksi dengan dataset validasi
        subtree_predictions = []
        for i in range(len(X_val)):
            prediction = self.predict_single_instance(X_val[i], node)
            subtree_predictions.append(prediction)
        subtree_accuracy = np.mean(np.array(subtree_predictions) == y_val)

        most_common = self.plurality_value(val_examples)
        leaf_accuracy = np.mean(y_val == most_common)

        # prune jika leaf accuracy >= subtree accuracy
        if leaf_accuracy >= subtree_accuracy:
            return TreeNode(is_leaf=True, classification=most_common, len_instances=len(val_examples))

        return node


    # Training model DTL dari training data
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X = np.array(X_train)
        y = np.array(y_train)

        self.classes = np.unique(y)
        n_features = X.shape[1]
        attributes = set(range(n_features))

        examples = np.column_stack((X, y))
        self.default_class = self.plurality_value(examples) # buat kasus unseen values di atribut

        self.tree = self.build_decision_tree(examples, attributes, parent_examples=None, depth=0)

        # Post pruning
        if self.pruning == 'rule':
            val_examples = np.column_stack((np.array(X_val), np.array(y_val)))
            self.post_pruning_by_rule(val_examples)
        elif self.pruning == 'reduced-error':
            val_examples = np.column_stack((np.array(X_val), np.array(y_val)))
            self.tree = self.reduced_error_pruning(self.tree, val_examples)

        return self

    # Predict satu instance dari dataset validasi berdasarkan model DTL
    def predict_single_instance(self, X_instance, node):

        if node.is_leaf:
            return node.classification

        attribute_value = X_instance[node.attribute]

        # atribut numerik
        if node.threshold is not None:
            if attribute_value <= node.threshold:
                return self.predict_single_instance(X_instance, node.branches['<='])
            else:
                return self.predict_single_instance(X_instance, node.branches['>'])

        # atribut kategorik
        else:
            if attribute_value in node.branches:
                return self.predict_single_instance(X_instance, node.branches[attribute_value])
            else:
                return self.default_class

    # Predict dataset validasi dengan DTL
    def predict(self, X_val):
        if self.tree is None:
            raise Exception("Model belum dilatih dengan fit().")

        arr_X = np.array(X_val)
        predictions = []

        if self.pruning == 'rule': # Untuk pruning by rule
            for i in range(len(arr_X)):
                prediction = self.default_class
                for rule in self.rules:
                    if rule.is_matched(arr_X[i]):
                        prediction = rule.classification
                        break
                predictions.append(prediction)

        else:
            for i in range(len(arr_X)):
                prediction = self.predict_single_instance(arr_X[i], self.tree)
                predictions.append(prediction)

        return np.array(predictions)

    # print Decision tree secara rekurisf
    def print_tree(self):
        if self.tree is None:
            raise Exception("Model belum dilatih dengan fit().")

        def print_node(node, depth):
            indent = "  " * depth
            if node.is_leaf:
                print(f"{indent}Leaf: Class = {node.classification} (len_instances = {node.len_instances})")
            else:
                if depth == 0:
                    prefix = "Root:"
                else:
                    prefix = "IntlNode:"

                print(f"{indent}{prefix} Attribute {node.attribute} (len_instances = {node.len_instances})")
                if node.threshold is not None:
                    print(f"{indent}  |->: <= {node.threshold}")
                    print_node(node.branches['<='], depth + 2)
                    print(f"{indent}  |->: > {node.threshold}")
                    print_node(node.branches['>'], depth + 2)
                else:
                    for value, subtree in node.branches.items():
                        print(f"{indent}  |->: value = {value}")
                        print_node(subtree, depth + 2)

        print_node(self.tree, 0)


    def save_model(self, file_path):
        model_data = {
            'tree': self.tree,
            'classes': self.classes,
            'default_class': self.default_class,
            'pruning': self.pruning,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'rules': self.rules
        }
        with open(file_path, 'wb') as file:
            pickle.dump(model_data, file)
        print(f"Model disimpan di {file_path}")

    def load_model(self, file_path):
        with open(file_path, 'rb') as file:
            model_data = pickle.load(file)
        
        self.tree = model_data['tree']
        self.classes = model_data['classes']
        self.default_class = model_data['default_class']
        self.pruning = model_data['pruning']
        self.max_depth = model_data['max_depth']
        self.min_samples_split = model_data['min_samples_split']
        self.min_samples_leaf = model_data['min_samples_leaf']
        self.rules = model_data['rules']

        print(f"Model berhasil di-load dari {file_path}")