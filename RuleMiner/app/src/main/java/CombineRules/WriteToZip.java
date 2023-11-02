package CombineRules;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Set;
import java.util.zip.*;

public class WriteToZip {

    public static ZipOutputStream createNewZip(String zipFileName) {
        ZipOutputStream zipOutputStream = null;
        try {

            // Delete existing zip if it exists
            File zipFile = new File(zipFileName);
            if (zipFile.exists()) {
                Files.delete(Paths.get(zipFileName));
            }

            // Create new zip file
            zipOutputStream = new ZipOutputStream(new FileOutputStream(zipFileName));


        } catch (IOException e) {
            e.printStackTrace();
        }
        return zipOutputStream;
    }

    public static void createFileAndWriteToZip(BufferedWriter bufferedWriter, String textToWrite) {
        try {
            // Create zip entry
            for (int i = 0; i<3; ++i) {
                bufferedWriter.write(textToWrite);
                bufferedWriter.newLine();
                bufferedWriter.flush();
            }
            // Close streams



        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) throws IOException {
        String zipName = "example.zip";

        createNewZip(zipName);
        ZipOutputStream zipOutputStream = new ZipOutputStream(new FileOutputStream(zipName, true));

        ZipEntry zipEntry = new ZipEntry("r0.txt");
        zipOutputStream.putNextEntry(zipEntry);

        // Create buffered writer and write text to file
        BufferedWriter bufferedWriter = new BufferedWriter(new OutputStreamWriter(zipOutputStream));
        createFileAndWriteToZip(bufferedWriter, "Hello!");
        zipOutputStream.closeEntry();
        bufferedWriter.close();

        zipOutputStream = new ZipOutputStream(new FileOutputStream(zipName, true));

        zipEntry = new ZipEntry("r1.txt");
        zipOutputStream.putNextEntry(zipEntry);

        // Create buffered writer and write text to file
        bufferedWriter = new BufferedWriter(new OutputStreamWriter(zipOutputStream));
        createFileAndWriteToZip(bufferedWriter, "World!");
        zipOutputStream.closeEntry();
        bufferedWriter.close();
    }
}