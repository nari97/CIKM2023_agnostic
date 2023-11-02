package CombineRules;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;

public class RuleParser {

    String filename = "";
    String model_name = "";
    String dataset_name = "";
    String relationship_delimiter = "";
    Map<String, List<Rule>> rules_by_predicate;
    List<Rule> rules;
    Map<String, String> id_to_relationship;
    String database_folder;
    
    public RuleParser(String filename, String database_folder, String model_name, String dataset_name, String relationship_delimiter) throws FileNotFoundException {
        this.filename = filename;
        this.model_name = model_name;
        this.dataset_name = dataset_name;
        this.relationship_delimiter = relationship_delimiter;
        this.database_folder = database_folder + dataset_name + "/";
        this.rules_by_predicate = new HashMap<>();
        this.rules = new ArrayList<>();
        this.id_to_relationship = new HashMap<>();
        if (database_folder != null)
            this.create_id_to_relationship();
    }

    public void create_id_to_relationship() throws FileNotFoundException {

        Scanner sc = new Scanner(new File(this.database_folder+"relation2id.txt"));
        sc.nextLine();

        while(sc.hasNextLine()){
            String line = sc.nextLine().strip();
            String[] splits = line.split(this.relationship_delimiter);
            this.id_to_relationship.put(splits[1], splits[0]);
        }
    }

    public void parse_rules_from_file(double beta) throws FileNotFoundException {
        Scanner sc = new Scanner(new File(this.filename));
        int num_lines = 0;

        while(sc.hasNextLine()){
            sc.nextLine();
            num_lines++;
        }

        sc.close();
        sc = new Scanner(new File(this.filename));
        for(int ctr = 0; ctr<15; ++ctr){
            sc.nextLine();
        }

        for(int ctr=15; ctr<num_lines-3; ++ctr){

            String line = sc.nextLine();
//            System.out.println(line);
            String splits[] = line.strip().split("\t");
            String functional_variable = splits[splits.length-1].replace("?", "");
            List<Atom> body_atoms = this.create_body_from_rule(splits[0]);
            Atom head_atom = this.create_head_from_rule(splits[0]);
            double hc = Double.parseDouble(splits[1]);
            double pca = Double.parseDouble(splits[3]);

            String relationship_id = head_atom.relationship;

            if (!this.rules_by_predicate.containsKey(relationship_id)){
                List<Rule> rules = new ArrayList<>();
                this.rules_by_predicate.put(relationship_id, rules);
            }

            Rule rule = new Rule(head_atom, body_atoms, hc, pca, functional_variable, beta);
            this.rules_by_predicate.get(relationship_id).add(rule);
            this.rules.add(rule);
        }

        for(String predicate: this.rules_by_predicate.keySet()){
            Collections.sort(this.rules_by_predicate.get(predicate), new Comparator<Rule>(){
                public int compare(Rule a, Rule b){
                    if (a.selectivity>b.selectivity){
                        return -1;
                    }
                    else if (a.selectivity<b.selectivity){
                        return 1;
                    }
                    else{
                        return 0;
                    }
                }
            });
        }
    }

    public List<Atom> create_body_from_rule(String rule){
        String splits[] = rule.split(" ");

        int body_atom_end_index = Arrays.asList(splits).indexOf("");

        List<Atom> body_atoms = new ArrayList<>();

        for(int i=0; i< body_atom_end_index; ++i){
            String atom_string = splits[i];
            String relationship_id = atom_string.substring(0, atom_string.indexOf("("));
            String variable1 = atom_string.substring(atom_string.indexOf("(")+1, atom_string.indexOf(","));
            String variable2 = atom_string.substring(atom_string.indexOf(",")+1, atom_string.indexOf(")"));
            String relationship_name = this.id_to_relationship.get(relationship_id);
            body_atoms.add(new Atom(relationship_id, variable1, variable2, relationship_name));
        }

        return body_atoms;
    }

    public Atom create_head_from_rule(String rule){
        String splits[] = rule.split(" ");
        String atom_string = splits[splits.length-1];
        String relationship_id = atom_string.substring(0, atom_string.indexOf("("));
        String variable1 = atom_string.substring(atom_string.indexOf("(")+1, atom_string.indexOf(","));
        String variable2 = atom_string.substring(atom_string.indexOf(",")+1, atom_string.indexOf(")"));
        String relationship_name = this.id_to_relationship.get(relationship_id);

        return new Atom(relationship_id, variable1, variable2, relationship_name);

    }
}
