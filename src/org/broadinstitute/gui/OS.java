/*
 * Copyright 2008-2009 The Broad Institute.  All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; version 2
 * of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package org.broadinstitute.gui;

import javax.swing.UIManager;

import org.apache.tools.ant.taskdefs.condition.Os;

/**
 * Utility class for properties having to do with the OS.
 * 
 * @author Joshua Gould
 */
public class OS {

    private OS() {
    }

    /**
     * Gets whether the operating system is Mac OS X.
     * 
     * @return <tt>true</tt> if the OS is Mac OS X, <tt>false</tt> otherwise.
     */
    public static boolean isMac() {
    	// Must check that we are running on both Mac and Unix to check for Mac OS X.
    	// Checking family "mac" allows for pre-OS X macs as well.
    	return Os.isFamily("mac") && Os.isFamily("unix");
    }

    /**
     * Gets whether the operating system is a Mac and a screen menu bar is in use.
     * 
     * @return <tt>true</tt> if the OS is a Mac, and a screen menu bar is in use, <tt>false</tt> otherwise.
     */
    public static boolean useScreenMenuBar() {
	return isMac() && "true".equals(System.getProperty("apple.laf.useScreenMenuBar"));
    }

    /**
     * Gets whether the operating system is Windows.
     * 
     * @return <tt>true</tt> if the OS is a Windows, <tt>false</tt> otherwise.
     */
    public static boolean isWindows() {
    	return Os.isFamily("windows");
    }

    public static boolean is64Bit() {
    	// Note: detects whether the JVM is 64-bit, not whether the underlying
    	// platform is 64-bit.  This seems to be about as good as we can do
    	// without resorting to OS-dependent arch checks.
	return "64".equals(System.getProperty("sun.arch.data.model"));
    }

    /**
     * Sets the look and feel to the system look and feel.
     * 
     */
    public static void setLookAndFeel() {
	if (!OS.isMac()) {
	    try {
		UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
	    } catch (Exception e) {
		System.err.println("Unable to set the look and feel.");
	    }
	}
    }
}
