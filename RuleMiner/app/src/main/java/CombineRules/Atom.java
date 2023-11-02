package CombineRules;

public class Atom {

    String relationship = "";
    String variable1 = "";
    String variable2 = "";
    String relationship_name = "";
    int placeholder = 1;
    public Atom(String relationship, String variable1, String variable2, String relationship_name){
        this.relationship = relationship;
        this.variable1 = variable1;
        this.variable2 = variable2;
        this.relationship_name = relationship_name;
    }

    public void set_placeholder(){
        if (this.variable1.equals("?g") || this.variable2.equals("?h")){
            this.placeholder = 0;
        }
    }

    public String id_print(){
        return this.relationship + "(" + this.variable1 + "," + this.variable2 + ")";
    }

    public String relationship_print(){
        return this.relationship_name + "(" + this.variable1 + "," + this.variable2 + ")";
    }

    public String neo4j_print(){
        String atom_to_neo4j = "(" + this.variable1 + ")-[:`" + this.relationship + "`]->(" + this.variable2 + ")";
        atom_to_neo4j = atom_to_neo4j.replace("?", "");
        return atom_to_neo4j;
    }


}
