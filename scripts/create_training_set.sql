create table training_set(object_id int, mjd double, passband int, flux_double, flux_err double, detected bool, PRIMARY KEY(object_id, passband, mjd));
.mode csv
.import training_set.csv training_set

create table training_set_meta(object_id int, ra double, decl double, gal_l double, gal_b double, ddf_bool, hostgal_specz double, hostgal_photoz double, hostgal_photoz_err double, distmod double, mwebv double, target int, PRIMARY KEY(object_id, target));
.mode csv
.import training_set_metadata.csv training_set_meta