

# Define your desired category lists assuming purely flood reduction (bamboo, plantations)
forest_flood_equivalent_classes = {
    'Open dry forest - Short',
    'Open dry forest - Tall (Woodland/Savanna)',
    'Disturbed broadleaved forest (Secondary Forest)',
    'Closed broadleaved forest (Primary Forest)',
    'Secondary Forest',
    'Fields and Secondary Forest',       # 50% Forest, 50% Ag
    'Bamboo and Secondary Forest',        # 50% Forest, 50% Ag (if you want some ag fraction)
    'Bamboo',
    'Plantation: Tree crops, shrub crops, sugar cane, banana',
    'Hardwood Plantation: Euculytus',
    'Hardwood Plantation: Mixed',
    'Hardwood Plantation: Mahoe',
    'Hardwood Plantation: Mahogany', 
}

afforestable_classes_including_agricultural = {
    'Fields: Herbaceous crops, fallow, cultivated vegetables',
    'Fields: Pasture,Human disturbed, grassland',
    'Fields and Secondary Forest',        # 50% Ag, 50% Forest
    'Bamboo and Fields',                  # 50% Ag, 50% Bamboo (if you consider bamboo here)
    'Fields  and Bamboo',                  # 50% Ag, 50% Bamboo
    'Fields or Secondary Forest/Pine Plantation',  # 50% Ag, 50% Secondary Forest/Pine
    'Fields: Bare Land',
    'Quarry',
    'Bauxite Extraction'
}

# Update the mixed land use fractions using the new keys.
mixed_land_use_fractions = {
    'Fields and Secondary Forest': {
         'forest_flood_equivalent_classes': 0.50,
         'afforestable_including_agriculture': 0.50
    },
    'Bamboo and Secondary Forest': {
         'forest_flood_equivalent_classes': 1
    },
    # If these classes are purely afforestable, you can allocate 100% to that category:
    'Bamboo and Fields': {
         'forest_flood_equivalent_classes': 0.50,
         'afforestable_including_agriculture': 0.50
    },
    'Fields  and Bamboo': {
         'forest_flood_equivalent_classes': 0.50,
         'afforestable_including_agriculture': 0.50
    },
    'Fields or Secondary Forest/Pine Plantation': {
         'forest_flood_equivalent_classes': 0.50,
         'afforestable_including_agriculture': 0.50
    }
}



def calculate_fractional_areas(row):
    """
    Calculate fractional areas for mixed land use types or assign the full area for single use types.
    """
    land_use_type = row['Classify']
    area = row.geometry.area

    # Apply fractional allocation if this is a mixed-use class
    if land_use_type in mixed_land_use_fractions:
        fractions = mixed_land_use_fractions[land_use_type]
        return {category: area * frac for category, frac in fractions.items()}

    # For single-use cases, check which category the land use type belongs to.
    elif land_use_type in forest_flood_equivalent_classes:
        return {'forest_flood_equivalent_classes': area}
    elif land_use_type in afforestable_classes_including_agricultural:
        return {'afforestable_including_agriculture': area}
    else:
        return {'other': area}




# -----------------------------------------------------------
# 0)  Class sets
# -----------------------------------------------------------
existing_forest_classes = {
    'Open dry forest - Short',
    'Open dry forest - Tall (Woodland/Savanna)',
    'Disturbed broadleaved forest (Secondary Forest)',
    'Closed broadleaved forest (Primary Forest)',
    'Secondary Forest',

}

reforestable_classes = {
    'Fields: Herbaceous crops, fallow, cultivated vegetables',
    'Fields: Pasture,Human disturbed, grassland',
    'Fields: Bare Land',
    'Quarry',
    'Bauxite Extraction'
}

treated_as_forest_classes = {
    'Bamboo',
    'Plantation: Tree crops, shrub crops, sugar cane, banana',
    'Hardwood Plantation: Euculytus',
    'Hardwood Plantation: Mixed',
    'Hardwood Plantation: Mahoe',
    'Hardwood Plantation: Mahogany',
}

mixed_land_use_fractions = {
    'Fields and Secondary Forest': {
         'existing_forest_classes': 0.50,
         'reforestable_classes': 0.50
    },
    'Bamboo and Secondary Forest': {
         'treated_as_forest_classes': 0.50,
         'existing_forest_classes': 0.50
    },
    'Bamboo and Fields': {
         'treated_as_forest_classes': 0.50,
         'reforestable_classes': 0.50
    },
    'Fields  and Bamboo': {
         'reforestable_classes': 0.50,
         'treated_as_forest_classes': 0.50
    },
    'Fields or Secondary Forest/Pine Plantation': {
         'reforestable_classes': 0.50,
         'treated_as_forest_classes': 0.50
    }
}











