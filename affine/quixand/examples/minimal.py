from affine import quixand as qs


def main():
	sbx = qs.Sandbox(template="python:3.11-slim", timeout=600, metadata={"user": "alice"})
	sbx.files.write("hello.txt", "hi!")
	print([f.path for f in sbx.files.ls(".")])
	out = sbx.run(["python", "-c", "print(2+2)"])
	print(out.text)
	execn = sbx.run_code("x=1\nx+=1\nprint(x)")
	print(execn.text)
	sbx.shutdown()


if __name__ == "__main__":
	main()


