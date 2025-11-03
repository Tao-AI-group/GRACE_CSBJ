class User:
    def __init__(self, name, user_id, gender, age, degree, num_children):
        self.name = name
        self.user_id = user_id
        self.gender = gender
        self.degree = degree
        self.age = age
        self.num_children = int(num_children)

    def to_dict(self):
        return {
            "name": self.name,
            "user_id": self.user_id,
            "gender": self.gender,
            "age": self.age,
            "degree": self.degree,
            "num_children": self.num_children
        }

    def __repr__(self):
        return (f"User(name={self.name}, user_id={self.user_id}, gender={self.gender}, "
                f"age={self.age}, degree={self.degree}, num_children={self.num_children})")

    # Generate a sentence describing the user's background.
    def generate_background_sentence(self):
        gender_pronoun = "he" if self.gender.lower() == "male" else "she"
        children_info = f"{self.num_children} children" if self.num_children > 0 else "no children"
        
        sentence = (
            f"{self.name} is a {self.age}-year-old {self.gender} with a degree in {self.degree}. "
            f"{gender_pronoun.capitalize()} has {children_info}."
        )
        
        return sentence