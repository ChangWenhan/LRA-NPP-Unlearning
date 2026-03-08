import pickle
import os
from collections import defaultdict

def _load_pkl_files(folder_path):
    """
    Helper function to load all .pkl files from a specified folder.
    Returns lists of loaded data and their corresponding filenames.
    This function should only be called once.
    """
    all_arrays = []
    file_names = []

    print(f"Scanning folder: {folder_path}\n")

    for filename in os.listdir(folder_path):
        if filename.endswith(".pkl"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, list):
                        all_arrays.append(data)
                        file_names.append(filename)
                        print(f"Successfully loaded '{filename}' with {len(data)} elements.")
                    else:
                        print(f"Warning: File '{filename}' does not contain a list. Skipping.")
            except Exception as e:
                print(f"Error reading '{filename}': {e}")
    return all_arrays, file_names

def find_largest_pairwise_intersection(all_arrays, file_names):
    """
    Finds the pair of lists with the largest intersection and prints the count of elements
    in that intersection, along with the names of the two files.
    Accepts pre-loaded lists and filenames.
    """
    if len(all_arrays) < 2:
        print("Not enough valid .pkl files containing lists (at least two needed) found to compare.")
        return

    max_intersection_size = -1
    best_pair = (None, None) # Stores (filename1, filename2)

    for i in range(len(all_arrays)):
        for j in range(i + 1, len(all_arrays)):
            set1 = set(all_arrays[i])
            set2 = set(all_arrays[j])

            intersection = set1.intersection(set2)
            current_intersection_size = len(intersection)

            if current_intersection_size > max_intersection_size:
                max_intersection_size = current_intersection_size
                best_pair = (file_names[i], file_names[j])

    print("\n" + "="*50)
    print("Largest Pairwise Intersection Analysis Results")
    print("="*50)
    print(f"Largest intersection found between: '{best_pair[0]}' and '{best_pair[1]}'")
    print(f"Number of common elements: {max_intersection_size}")
    print("="*50 + "\n")

def analyze_pkl_arrays(all_arrays, file_names):
    """
    Finds the common elements across all loaded lists and prints the percentage of
    these common elements within each original list.
    Accepts pre-loaded lists and filenames.
    """
    if not all_arrays:
        print("No valid .pkl files containing lists were found.")
        return

    if len(all_arrays) == 1:
        common_elements = set(all_arrays[0])
        print("\nOnly one array found. Its elements are considered the common elements.")
    else:
        common_elements = set(all_arrays[0])
        for i in range(1, len(all_arrays)):
            common_elements.intersection_update(set(all_arrays[i]))

    print("\n" + "="*50)
    print("Common Elements Across All Arrays Analysis Results")
    print("="*50)
    print(f"Common elements across all arrays: {list(common_elements)}")

    print("\nPercentage of common elements in each array:")
    for i, arr in enumerate(all_arrays):
        if len(arr) == 0:
            print(f"  {file_names[i]}: Array is empty, cannot calculate percentage.")
            continue

        count_common_in_array = 0
        for element in arr:
            if element in common_elements:
                count_common_in_array += 1

        percentage = (count_common_in_array / len(arr)) * 100
        print(f"  {file_names[i]}: {percentage:.2f}%")
    print("="*50 + "\n")

def find_element_most_present_in_lists(all_arrays, file_names):
    """
    Finds the element that appears in the maximum number of distinct lists
    and prints that element along with the count of lists it appeared in.
    Accepts pre-loaded lists and filenames.
    """
    if not all_arrays:
        print("No valid .pkl files containing lists were found.")
        return

    # Use a defaultdict to store elements and the set of list indices they appear in
    element_to_list_indices = defaultdict(set)

    for list_idx, current_list in enumerate(all_arrays):
        # Convert the current list to a set to handle duplicate elements within the same list
        # correctly (i.e., an element appearing multiple times in one list still counts as 1 list)
        unique_elements_in_current_list = set(current_list)
        for element in unique_elements_in_current_list:
            element_to_list_indices[element].add(list_idx)

    if not element_to_list_indices:
        print("No elements found across any of the loaded lists.")
        return

    max_lists_count = 0
    most_frequent_elements = [] # Store elements that appear in the max number of lists

    for element, list_indices_set in element_to_list_indices.items():
        current_lists_count = len(list_indices_set)
        if current_lists_count > max_lists_count:
            max_lists_count = current_lists_count
            most_frequent_elements = [element] # Start a new list if a new max is found
        elif current_lists_count == max_lists_count:
            most_frequent_elements.append(element) # Add to existing list if count is equal to max

    print("\n" + "="*50)
    print("Element Appearing in Most Lists Analysis Results")
    print("="*50)
    print(f"The element(s) appearing in the most lists: {most_frequent_elements}")
    print("length is ", len(most_frequent_elements))
    print(f"It/They appeared in a total of {max_lists_count} lists.")
    print("="*50 + "\n")


if __name__ == "__main__":
    # IMPORTANT: Replace 'path/to/your/pkl_folder' with the actual path to your folder.
    # For example:
    # folder_to_analyze = "/Users/yourusername/Documents/my_data_files"
    # folder_to_analyze = "C:\\Users\\yourusername\\Desktop\\data"
    # folder_to_analyze = "./data_files" # If the folder is the same directory as your script

    folder_to_analyze = "neuron/mufac" # <--- YOU MUST CHANGE THIS LINE TO YOUR FOLDER PATH

    if not os.path.isdir(folder_to_analyze):
        print(f"\nError: The specified folder '{folder_to_analyze}' does not exist.")
        print("Please update the 'folder_to_analyze' variable with the correct path to your .pkl files.")
    else:
        # Load all PKL files once at the start
        loaded_arrays, loaded_file_names = _load_pkl_files(folder_to_analyze)

        # Check if any valid files were loaded before proceeding to the menu
        if not loaded_arrays:
            print("No valid .pkl files found. Exiting.")
        else:
            while True:
                print("Choose an analysis option:")
                print("1. Find largest pairwise intersection")
                print("2. Analyze common elements across all arrays and their percentages")
                print("3. Find element appearing in most lists")
                print("4. Perform all analyses")
                print("5. Exit")
                choice = input("Enter your choice (1/2/3/4/5): ").strip()

                if choice == '1':
                    find_largest_pairwise_intersection(loaded_arrays, loaded_file_names)
                elif choice == '2':
                    analyze_pkl_arrays(loaded_arrays, loaded_file_names)
                elif choice == '3':
                    find_element_most_present_in_lists(loaded_arrays, loaded_file_names)
                elif choice == '4':
                    find_largest_pairwise_intersection(loaded_arrays, loaded_file_names)
                    analyze_pkl_arrays(loaded_arrays, loaded_file_names)
                    find_element_most_present_in_lists(loaded_arrays, loaded_file_names)
                elif choice == '5':
                    print("Exiting program.")
                    break
                else:
                    print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")