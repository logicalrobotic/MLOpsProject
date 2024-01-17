import pandas as pd
from os.path import dirname as up
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, power_transform
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class Raw2Clean:
    def __init__(self):
        self.headers = ['time_left', 'ct_score', 't_score', 'map', 'bomb_planted', 
               'ct_health', 't_health', 'ct_armor', 't_armor', 'ct_money', 
               't_money', 'ct_helmets', 't_helmets', 'ct_defuse_kits', 'ct_players_alive', 
               't_players_alive', 'ct_weapon_ak47', 't_weapon_ak47', 'ct_weapon_aug', 't_weapon_aug', 
               'ct_weapon_awp', 't_weapon_awp', 'ct_weapon_bizon', 't_weapon_bizon', 'ct_weapon_cz75auto', 
               't_weapon_cz75auto', 'ct_weapon_elite', 't_weapon_elite', 'ct_weapon_famas', 't_weapon_famas', 
               'ct_weapon_g3sg1', 't_weapon_g3sg1', 'ct_weapon_galilar', 't_weapon_galilar', 'ct_weapon_glock', 
               't_weapon_glock', 'ct_weapon_m249', 't_weapon_m249', 'ct_weapon_m4a1s', 't_weapon_m4a1s', 
               'ct_weapon_m4a4', 't_weapon_m4a4', 'ct_weapon_mac10', 't_weapon_mac10', 'ct_weapon_mag7', 
               't_weapon_mag7', 'ct_weapon_mp5sd', 't_weapon_mp5sd', 'ct_weapon_mp7', 't_weapon_mp7', 
               'ct_weapon_mp9', 't_weapon_mp9', 'ct_weapon_negev', 't_weapon_negev', 'ct_weapon_nova', 
               't_weapon_nova', 'ct_weapon_p90', 't_weapon_p90', 'ct_weapon_r8revolver', 't_weapon_r8revolver', 
               'ct_weapon_sawedoff', 't_weapon_sawedoff', 'ct_weapon_scar20', 't_weapon_scar20', 'ct_weapon_sg553', 
               't_weapon_sg553', 'ct_weapon_ssg08', 't_weapon_ssg08', 'ct_weapon_ump45', 't_weapon_ump45', 
               'ct_weapon_xm1014', 't_weapon_xm1014', 'ct_weapon_deagle', 't_weapon_deagle', 'ct_weapon_fiveseven', 
               't_weapon_fiveseven', 'ct_weapon_usps', 't_weapon_usps', 'ct_weapon_p250', 't_weapon_p250', 
               'ct_weapon_p2000', 't_weapon_p2000', 'ct_weapon_tec9', 't_weapon_tec9', 'ct_grenade_hegrenade', 
               't_grenade_hegrenade', 'ct_grenade_flashbang', 't_grenade_flashbang', 'ct_grenade_smokegrenade', 't_grenade_smokegrenade', 
               'ct_grenade_incendiarygrenade', 't_grenade_incendiarygrenade', 'ct_grenade_molotovgrenade', 't_grenade_molotovgrenade', 'ct_grenade_decoygrenade', 
               't_grenade_decoygrenade', 'round_winner']

    def remove_grenades(self, df):
        """
        Removes grenade columns from the dataframe
        """
        # Get the columns with grenade info
        cols_grenade = df.columns[df.columns.str.contains('grenade')]

        # Drop the columns
        df = df.drop(cols_grenade, axis=1)

        return df

    def encode_targets(self, y, possible_outcomes):
        encoder = LabelEncoder()
        encoder.fit(possible_outcomes)
        y_encoded = encoder.transform(y)
        return y_encoded

    def encode_inputs(self, X, object_cols, map_list, bomb_list):
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False, categories=[map_list, bomb_list])
        X_encoded = pd.DataFrame(ohe.fit_transform(X[object_cols]))
        X_encoded.columns = ohe.get_feature_names_out(object_cols)
        X_encoded.index = X.index
        return X_encoded

        
    def yeo_johnson(self, series):
        arr = np.array(series).reshape(-1, 1)
        return power_transform(arr, method='yeo-johnson')
    
    def clean_data(self, values:list):
        # create a dataframe using the data and headers
        df = pd.DataFrame([values], columns=self.headers)

        # Split X and y
        y = df.round_winner
        X = df.drop(['round_winner'], axis=1)

        # Drop columns with grenade info
        #X = self.remove_grenades(X)

        #print(f"Total number of samples: {len(X)}")
        #print(X.head())

        # Use OH encoder to encode predictors
        object_cols = ['map', 'bomb_planted']

        
        map_list = ['de_cache', 'de_dust2', 'de_inferno', 'de_mirage', 'de_nuke', 'de_overpass', 'de_train', 'de_vertigo']
        bomb_list = [False, True]
        X_encoded = self.encode_inputs(X, object_cols, map_list, bomb_list)
        numerical_X = X.drop(object_cols, axis=1)
        X = pd.concat([numerical_X, X_encoded], axis=1)

        # Use label encoder to encode targets
        y = self.encode_targets(y, ['CT', 'T'])

        # Make data more Gaussian-like
        cols = ['time_left', 'ct_money', 't_money', 'ct_health',
                't_health', 'ct_armor', 't_armor', 'ct_helmets', 't_helmets',
                'ct_defuse_kits', 'ct_players_alive', 't_players_alive']
        for col in cols:
            X[col] = self.yeo_johnson(X[col])

        #print(f"Total number of samples: {len(X)}")
        #print(X.head())

        X_test_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y, dtype=torch.long)

        test_dataset = CustomDataset(X_test_tensor, y_test_tensor)

        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        return test_dataloader
if __name__ == "__main__":
    # Load the data
    import sys
    sys.path.append("./MLOpsProject")
    from predict_model import predict
    r2c = Raw2Clean()
    data_raw = [114.97, 2.0, 0.0, 'de_dust2', False, 
                500.0, 500.0, 496.0, 500.0, 2200.0, 
                1000.0, 4.0, 5.0, 2.0, 5.0, 
                5.0, 0.0, 5.0, 0.0, 0.0, 
                1.0, 0.0, 0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0, 1.0, 0.0, 
                0.0, 0.0, 0.0, 0.0, 0.0, 
                5.0, 0.0, 0.0, 0.0, 0.0, 
                2.0, 0.0, 0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0, 0.0, 0.0, 
                0.0, 1.0, 0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0, 0.0, 0.0, 
                0.0, 4.0, 0.0, 0.0, 0.0, 
                1.0, 0.0, 0.0, 0.0, 1.0, 
                0.0, 4.0, 5.0, 4.0, 5.0, 
                2.0, 0.0, 0.0, 1.0, 0.0, 
                1.0, 'T']
    
    dataloader = r2c.clean_data(data_raw)

    model_path = "./models/trained_model.pt"
    test_path = "./processed/test_loader.pth"
    print("Loading data from: ", test_path)
    test_loader = torch.load(test_path)
    
    predicted = predict(model_path, dataloader)
    winner_dict = {0: 'CT', 1: 'T'}
    print("Predicted winner:", winner_dict[predicted[0].argmax().item()], "with probability:", predicted[0].max().item()*100, "%")
