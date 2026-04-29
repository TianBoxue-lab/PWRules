from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def read_smiles_list_from_smi(input_path):
    smiles_list = []

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    smiles = line.split()[0]
                    smiles_list.append(smiles)

    except FileNotFoundError:
        print(f"Error: File {input_path} does not exist")
        return []
    except UnicodeDecodeError:
        try:
            with open(input_path, 'r', encoding='gbk') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        smiles = line.split()[0]
                        smiles_list.append(smiles)
        except Exception as e:
            print(f"Error reading file: {e}")
            return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

    print(f"Successfully read {len(smiles_list)} SMILES from {input_path}")
    return smiles_list


def read_smiles_dict_from_smi(input_path):
    smiles_dict = {}

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        smiles = parts[0]
                        name = parts[1]
                        smiles_dict[smiles] = name
                    else:
                        smiles = parts[0]
                        smiles_dict[smiles] = smiles

    except Exception as e:
        print(f"Error reading file: {e}")
        return {}

    print(f"Successfully read {len(smiles_dict)} SMILES from {input_path}")
    return smiles_dict


def get_longest_molecule_component(input_smiles, use_atom_count=True):
    if '.' not in input_smiles:
        return input_smiles

    components = input_smiles.split('.')

    if use_atom_count:
        max_atoms = -1
        largest_component = components[0]

        for comp in components:
            try:
                mol = Chem.MolFromSmiles(comp)
                if mol:
                    num_atoms = mol.GetNumAtoms()
                    if num_atoms > max_atoms:
                        max_atoms = num_atoms
                        largest_component = comp
            except:
                continue

        return largest_component
    else:
        return max(components, key=len)
