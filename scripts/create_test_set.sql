create table test_set(object_id int, mjd double, passband int, flux double, flux_err double, detected bool, PRIMARY KEY(object_id, passband, mjd));
.mode csv
.import test_set.csv test_set

create table test_set_meta(object_id int, ra double, decl double, gal_l double, gal_b double, ddf bool, hostgal_specz double, hostgal_photoz double, hostgal_photoz_err double, distmod double, mwebv double, PRIMARY KEY(object_id));
.mode csv
.import test_set_metadata.csv test_set_meta