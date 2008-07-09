/*
 The Broad Institute
 SOFTWARE COPYRIGHT NOTICE AGREEMENT
 This software and its documentation are copyright (2003-2006) by the
 Broad Institute/Massachusetts Institute of Technology. All rights are
 reserved.

 This software is supplied without any warranty or guaranteed support
 whatsoever. Neither the Broad Institute nor MIT can be responsible for its
 use, misuse, or functionality.
 */

package org.genepattern.modules.hcl;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.genepattern.io.DatasetParser;
import org.genepattern.io.IOUtil;
import org.genepattern.io.stanford.StanfordTxtWriter;
import org.genepattern.matrix.Dataset;
import org.genepattern.module.AnalysisUtil;
import org.genepattern.module.ExecutableWrapper;

public class RunCluster extends ExecutableWrapper {

    public RunCluster(String[] args) {
	super(args);
    }

    public static void main(String[] args) {
	new RunCluster(args);
    }

    @Override
    protected String[] createNewArgs() {

	String inputFileName = args[1];
	DatasetParser reader = AnalysisUtil.getDatasetParser(inputFileName);
	Dataset dataset = AnalysisUtil.readDataset(reader, inputFileName);
	StanfordTxtWriter writer = new StanfordTxtWriter();
	String txtFileName = IOUtil.getBaseFileName(inputFileName) + ".txt";
	BufferedOutputStream os = null;
	try {
	    os = new BufferedOutputStream(new FileOutputStream(txtFileName));
	    writer.write(dataset, os);
	} catch (IOException x) {
	    System.err.println("Unable to create temporary file.");
	    System.exit(1);
	} finally {
	    if (os != null) {
		try {
		    os.close();
		} catch (IOException e) {
		}
	    }
	    new File(txtFileName).deleteOnExit();
	}

	args[1] = txtFileName;
	List<String> newArgs = new ArrayList<String>(Arrays.asList(args));
	newArgs.add(0, executable);

	for (int i = 0; i < newArgs.size(); i++) {
	    String arg = newArgs.get(i);
	    if (arg.equals("mean.row")) {
		newArgs.remove(i); // median.row median.col
		newArgs.add("-cg");
		newArgs.add("a");
		i--;
	    } else if (arg.equals("median.row")) {
		newArgs.remove(i);
		newArgs.add("-cg");
		newArgs.add("m");
		i--;
	    }
	    if (arg.equals("mean.column")) {
		newArgs.remove(i);
		newArgs.add("-ca");
		newArgs.add("a");
		i--;
	    } else if (arg.equals("median.column")) {
		newArgs.remove(i);
		newArgs.add("-ca");
		newArgs.add("m");
		i--;
	    }
	}

	return newArgs.toArray(new String[0]);

    }
}