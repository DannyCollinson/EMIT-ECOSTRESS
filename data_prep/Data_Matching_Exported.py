# %%
import os
# import sys
import pickle

import numpy as np
import xarray as xr
# import pandas as pd
import matplotlib.pyplot as plt

# import rioxarray
from osgeo import osr
osr.UseExceptions()

project_path = 'C:\\Users\\danny\\Documents_Local\\Caltech_2023-2024\\CS101'

def join_path(relative_path: str) -> str:
    return os.path.join(project_path, relative_path)

def pickle_load(relative_path: str):  # -> pickled_file_contents
    return pickle.load(open(join_path(relative_path), 'rb'))
    
# modules_path = 'VITALS\\user_contributed\\modules'
# if os.path.join(project_path, modules_path) not in sys.path:
    # sys.path.append(join_path(modules_path))
# from emit_tools import emit_xarray

# %%
# emit_002_path = join_path(
#     'Data\\Raw_Data\\EMIT_L2A_RFL_001_20230728T214106_2320914_002.nc'
# )

# emit_003_path = join_path(
#     'Data\\Raw_Data\\EMIT_L2A_RFL_001_20230728T214118_2320914_003.nc'
# )

# emit_002_clean_path = join_path('Data\\Clean_Data\\emit_002.nc')
# emit_003_clean_path = join_path('Data\\Clean_Data\\emit_003.nc')

# eco_slt_path = join_path(
#     'Data\\Raw_Data\\'
#     'ECOv002_L2T_LSTE_28691_004_11SLT_20230728T214058_0710_01_LST.tif'
# )
# eco_slu_path = join_path(
#     'Data\\Raw_Data\\'
#     'ECOv002_L2T_LSTE_28691_004_11SLU_20230728T214058_0710_01_LST.tif'
# )
# eco_smt_path = join_path(
#     'Data\\Raw_Data\\'
#     'ECOv002_L2T_LSTE_28691_004_11SMT_20230728T214058_0710_01_LST.tif'
# )

emit_clean_path = join_path('Data\\Clean_Data\\emit.nc')

eco_clean_path = join_path('Data\\Clean_Data\\ecostress.nc')

emit_yx_path = join_path('Data\\Clean_Data\\emit_yx.nc')

eco_on_emit_path = join_path('Data\\Matched_Data\\eco_on_emit_yx.nc')

data_yx_path = join_path('Data\\Matched_Data\\data_yx.nc')

decomposed_path = join_path('Data\\Decomposed_Data\\(__).pkl')


# %% [markdown]
# # Cleaning EMIT datasets

# %%
emit_002 = emit_xarray(emit_002_path, ortho=True)
emit_003 = emit_xarray(emit_003_path, ortho=True)

# %%
emit_002 = emit_002.reset_coords('elev')
emit_003 = emit_003.reset_coords('elev')

# %%
for (num_cur, emit_cur) in [('002', emit_002), ('003', emit_003)]:
    # coords
    wavelengths = emit_cur.coords['wavelengths'][
        emit_cur.coords['good_wavelengths'].astype(bool)
    ]
    
    fwhm = emit_cur.coords['fwhm'][
        emit_cur.coords['good_wavelengths'].astype(bool)
    ]
    
    good_wavelengths = emit_cur.coords['good_wavelengths'][
        emit_cur.coords['good_wavelengths'].astype(bool)
    ]

    lat = emit_cur.coords['latitude']
    lon = emit_cur.coords['longitude']
    
    spatial_ref = emit_cur.coords['spatial_ref']
    
    coords ={
        'wavelengths': wavelengths,
        'fwhm': fwhm,
        'good_wavelengths': good_wavelengths,
        'latitude': lat,
        'longitude': lon,
        'spatial_ref': spatial_ref,
    }


    # data_vars
    reflectance = np.clip(
        emit_cur.variables['reflectance'][
            :, :, emit_cur.variables['reflectance'].sum(axis=(0,1)) > 0
        ],
        0,
        1,
    )

    elev = emit_cur.variables['elev']

    data_vars = {'reflectance':reflectance, 'elev':elev}


    # build clean dataset
    emit_cur_clean = xr.Dataset(data_vars, coords)
    
    emit_cur_clean = emit_cur_clean.assign_attrs(emit_cur.attrs)

    emit_cur_clean.to_netcdf(
        os.path.join(project_path, f'Data\\Clean_Data\\emit_{num_cur}.nc')
    )

# %% [markdown]
# # Combining EMIT datasets

# %%
emit_002 = xr.load_dataset(emit_002_clean_path)
emit_003 = xr.load_dataset(emit_003_clean_path)

# %%
# coords
wavelengths = emit_002.coords['wavelengths']
fwhm = emit_002.coords['fwhm']
good_wavelengths = emit_002.coords['good_wavelengths']
spatial_ref = emit_002.coords['spatial_ref']

lat2 = emit_002.coords['latitude'].values
lat3 = emit_003.coords['latitude'].values
lon2 = emit_002.coords['longitude'].values
lon3 = emit_003.coords['longitude'].values

lat_concat = np.concatenate([lat2, lat3], axis=0)
lon_concat = np.concatenate([lon2, lon3], axis=0)

lat = np.sort(lat_concat)#[::-1]
lon = np.sort(lon_concat)

lat_order = np.argsort(np.argsort(lat_concat))#[::-1]
lon_order = np.argsort(np.argsort(lon_concat))

coords = {
    'wavelengths': wavelengths,
    'fwhm': fwhm,
    'good_wavelengths': good_wavelengths,
    'spatial_ref': spatial_ref,
    'latitude': lat,
    'longitude': lon
}

# %%
ref2 = emit_002.variables['reflectance'].values
ref3 = emit_003.variables['reflectance'].values

elev2 = emit_002.variables['elev'].values
elev3 = emit_003.variables['elev'].values

# %%
ref = np.empty((len(lat), len(lon), ref2.shape[2])) + np.nan
elev = np.empty((len(lat), len(lon))) + np.nan

for i in range(len(lat2)):
    for j in range(len(lon2)):
        ref[lat_order[i], lon_order[j], :] = ref2[i, j, :]
        elev[lat_order[i], lon_order[j]] = elev2[i, j]

for i in range(len(lat3)):
    for j in range(len(lon3)):
        ref[lat_order[len(lat2) + i], lon_order[len(lon2) + j], :] = (
            ref3[i, j, :]
        )
        elev[lat_order[len(lat2) + i], lon_order[len(lon2) + j]] = elev3[i, j]

data_vars = {
    'reflectance':(('latitude', 'longitude', 'wavelengths'), ref),
    'elev':(('latitude', 'longitude'), elev),
}

emit_clean = xr.Dataset(data_vars, coords)

# %%
emit_clean = emit_clean.set_coords('elev')
emit_clean['elev'].attrs = {'long_name': 'Surface Elevation', 'units': 'm'}
emit_clean['latitude'].attrs = {
    'long_name': 'Latitude (WGS-84)', 'units': 'degrees north'
}
emit_clean['longitude'].attrs = {
    'long_name': 'Latitude (WGS-84)', 'units': 'degrees east'
}
emit_clean = emit_clean.assign_attrs(emit_002.attrs)

# %%
emit_clean.to_netcdf(emit_clean_path)

# %%
# for visualization

elev = np.empty((len(lat), len(lon))) + np.nan

for i in range(len(lat2)):
    for j in range(len(lon2)):
        elev[lat_order[i], lon_order[j]] = elev2[i, j]

for i in range(len(lat3)):
    for j in range(len(lon3)):
        elev[
            lat_order[len(lat2) + i],
            lon_order[len(lon2) + j]
        ] = elev3[i, j]

plt.show(plt.matshow(elev))
plt.matshow(elev3)
plt.matshow(elev2)

# %% [markdown]
# # Combining ECOSTRESS datasets

# %%
eco_slt = rioxarray.open_rasterio(eco_slt_path)
eco_slu = rioxarray.open_rasterio(eco_slu_path)
eco_smt = rioxarray.open_rasterio(eco_smt_path)

# %%
eco_slt_xr = eco_slt.to_dataset('band').rename({1:'LSTE'})
eco_slu_xr = eco_slu.to_dataset('band').rename({1:'LSTE'})
eco_smt_xr = eco_smt.to_dataset('band').rename({1:'LSTE'})
eco_df = pd.concat(
    [
    eco_slt_xr.to_dataframe(),
    eco_slu_xr.to_dataframe(),
    eco_smt_xr.to_dataframe(),
    ]
)

# %%
eco_unique = eco_df.reset_index().drop_duplicates(subset=['y', 'x'])
eco = eco_unique.set_index(['y', 'x']).to_xarray()
eco = eco.assign_attrs(eco_slt.attrs)
eco = eco.reset_coords('spatial_ref', drop=True)
eco = eco.assign_coords({'spatial_ref':eco_slt_xr.variables['spatial_ref']})

# %%
eco.to_netcdf(eco_clean_path)

# %% [markdown]
# # Aligning EMIT and ECOSTRESS data

# %%
# load emit
emit = xr.open_dataset(emit_clean_path)
emit = emit.reset_coords('elev')

# load ecostress
eco = xr.open_dataset(eco_clean_path)

# %%
# get the coordinate systems
old_cs = osr.SpatialReference()
old_cs.ImportFromWkt(emit['spatial_ref'].attrs['crs_wkt'])

new_cs = osr.SpatialReference()
new_cs.ImportFromWkt(eco['spatial_ref'].attrs['crs_wkt'])

transform = osr.CoordinateTransformation(old_cs, new_cs)

emit_lat = emit['latitude'].values
emit_lon = emit['longitude'].values

# %%
emit_y = np.empty(len(emit_lat)) + np.nan
emit_x = np.empty(len(emit_lon)) + np.nan

for y_ind, lat_val in enumerate(emit_lat):
    emit_y[y_ind] = transform.TransformPoint(
        lat_val,
        emit_lon[int(y_ind / 0.81)],
    )[1]
    
for x_ind, lon_val in enumerate(emit_lon):
    emit_x[x_ind] = transform.TransformPoint(
        emit_lat[int(x_ind * 0.8)],
        lon_val,
    )[0]

(
    (
        np.abs(np.argsort(emit_y) - np.arange(len(emit_y))).sum(),
        np.abs(np.argsort(emit_x) - np.arange(len(emit_x))).sum()
    ),
    (
        len(emit_y) - len(np.unique(emit_y)),
        len(emit_x) - len(np.unique(emit_x)),

    ),
)

# %%
# coords
wavelengths = emit.coords['wavelengths']
fwhm = emit.coords['fwhm']
good_wavelengths = emit.coords['good_wavelengths']
spatial_ref = emit.coords['spatial_ref']

y = np.sort(emit_y)
x = np.sort(emit_x)

coords = {
    'wavelengths': wavelengths,
    'fwhm': fwhm,
    'spatial_ref': spatial_ref,
    'y': y,
    'x': x,
}

# %%
# variables
ref_old = emit['reflectance'].values
elev_old = emit['elev'].values

y_order = np.argsort(np.argsort(emit_y))
x_order = np.argsort(np.argsort(emit_x))

ref = np.empty_like(ref_old) + np.nan
elev = np.empty_like(elev_old) + np.nan

# %%
for i in range(len(y)):
    for j in range(len(x)):
        ref[y_order[i], x_order[j], :] = ref_old[i, j, :]
        elev[y_order[i], x_order[j]] = elev_old[i, j]

data_vars = {
    'reflectance':(('y', 'x', 'wavelengths'), ref),
    'elev':(('y', 'x'), elev),
}

# %%
emit_yx = xr.Dataset(data_vars, coords)
emit_yx = emit_yx.assign_attrs(emit.attrs)

# %%
emit_yx.to_netcdf(emit_yx_path)

# %%
emit_yx = xr.open_dataset(emit_yx_path)

eco = xr.open_dataset(eco_clean_path)

# %%
interp_eco = eco.interp_like(emit_yx, assume_sorted=True)

# %%
interp_eco.to_netcdf(eco_on_emit_path)

# %% [markdown]
# # Processing data

# %%
emit_yx = xr.open_dataset(emit_yx_path)
eco_on_emit = xr.open_dataset(eco_on_emit_path)

# %%
data_yx = emit_yx.assign({'LSTE': eco_on_emit['LSTE']})

# %%
lste = data_yx['LSTE'].values
ref_sum = data_yx['reflectance'].values.sum(axis=2)
elev = data_yx['elev'].values

# %%
nan_mask = np.isnan(lste) + np.isnan(ref_sum) + np.isnan(elev)
nan_mask = ~nan_mask.astype(bool)
data_yx = data_yx.assign_coords({'good_coords': (('y', 'x'), nan_mask)})

# %%
data_yx.to_netcdf(data_yx_path)

# %%
data_yx = xr.open_dataset(data_yx_path)

# %%
good_coords = data_yx['good_coords'].values
y_filter = good_coords.sum(axis=1) > 0
x_filter = good_coords.sum(axis=0) > 0
good_coords = good_coords[y_filter, :]
good_coords = good_coords[:, x_filter]

y = data_yx['y'].values[y_filter]
x = data_yx['x'].values[x_filter]

ref = data_yx['reflectance'].values
ref = ref[y_filter, :, :]
ref = ref[:, x_filter, :]

elev = data_yx['elev'].values
elev = elev[y_filter, :]
elev = elev[:, x_filter]

lste = data_yx['LSTE'].values
lste = lste[y_filter, :]
lste = lste[:, x_filter]

# %%
plt.matshow(lste[::-1, :])
plt.matshow(elev[::-1, :])
plt.matshow(ref[::-1, :, 100])

# %%
ref_good = ref[good_coords, :]
elev_good = elev[good_coords]
lste_good = lste[good_coords]

# %%
pickle_list = [
    ('good_coords', good_coords),
    ('y', y),
    ('x', x),
    ('reflectance', ref),
    ('reflectance_filtered', ref_good),
    ('elevation', elev),
    ('elevation_filtered', elev_good),
    ('LSTE', lste),
    ('LSTE_filtered', lste_good),
]

# %%
for filename, data in pickle_list:
    pickle.dump(data, open(decomposed_path.replace('(__)', filename), 'wb'))

# %%
# for visualization
plt.matshow(~np.isnan(lste))
plt.matshow(~np.isnan(ref_sum))
plt.matshow(~np.isnan(elev))
plt.matshow(nan_mask)
plt.matshow(ref[:,:,100])

# %%
plt.imshow(ref_good, aspect='auto')


