# graphdb.py


import ladybug


class LadybugGraphDB:
	def __init__(self, db_path: str):
		self.db = ladybug.Database(db_path)
		self.conn = ladybug.Connection(self.db)


		
	

	def add_triplet(self, subject: str, relation: str, object: str, type: str = "entity") -> None:
		self.conn.add(subject, relation, object, type=type)