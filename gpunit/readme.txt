Setup ... add these lines to your ~/.profile 
  # install ant
  export ANT_HOME=/path/to/ant
  export PATH=$PATH:${ANT_HOME}/bin

  export GPUNIT_HOME=/path/to/GpUnit

To run the tests ...

  ant -f ${GPUNIT_HOME}/build.xml \
    -Dgpunit.diffStripTrailingCR="--strip-trailing-cr" \
    -Dgp.url="https://genepattern.broadinstitute.org" \
    -Dgp.user="your_username" \
    -Dgp.password="your_password" \
    -Dgpunit.testfolder=`pwd` \
  gpunit

To view the report ...
  open reports/current/html/index.html


To run GpUnits locally:

ant -f ${GPUNIT_HOME}/build.xml -Dgpunit.diffStripTrailingCR="--strip-trailing-cr" -Dgp.host="127.0.0.1" -Dgp.url="http://127.0.0.1:8080" -Dgp.user="edjuaro@gmail.com" -Dgp.password="" -Dgpunit.testfolder=`pwd` gpunit

