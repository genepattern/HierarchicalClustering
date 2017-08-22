  # To run the gp-unit tests, execute them from the gp-unit dir but with
# this gpunit.properties file

cd ../../../util/gp-unit/
ant -Dgpunit.properties=../../HierarchicalClustering/gp-unit/v6/gpunit.properties gpunit
