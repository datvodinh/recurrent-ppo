import subprocess
import json
from git import Repo
from tests.CheckEnv import check_pytest
from tests.CheckGraphics import check_graphics
from tests.CheckPlayer import check_agent


def get_changed_files():
    repo_path = ""
    repo = Repo(repo_path)
    latest_commit = repo.head.commit

    commit = f"{latest_commit.hexsha}"
    api_path = f"https://api.github.com/repos/ngoxuanphong/ENV/commits/{commit}"
    command = f"curl -s {api_path}"
    output = subprocess.check_output(
        command, shell=True, stderr=subprocess.STDOUT
    ).decode("utf-8")

    output = json.loads(output)

    print("Last commit: ", latest_commit.hexsha)
    print("API:", api_path)
    changed_files = []
    for file in output["files"]:
        print(file["filename"])
        changed_files.append(file["filename"])

    return changed_files


def test_print_name():
    changed_files = get_changed_files()
    for file in changed_files:
        if "src/Base/" in file and "/env.py" in file:
            env_name = file.replace("src/Base/", "").replace("/env.py", "")
            print(env_name, "checking...")
            check_env, list_bug = check_pytest(env_name)
            if check_env == False:
                print("ENV:", env_name, "FALSE:", list_bug)
                assert False
            else:
                print("ENV:", env_name, "TRUE")
                assert True

        if "src/Agent/" in file and "/Agent_player.py" in file:
            agent_name = file.replace("src/Agent/", "").replace("/Agent_player.py", "")
            print(agent_name, "checking...")
            bool_check_agent, list_bug = check_agent(agent_name)
            if bool_check_agent == False:
                print("Agent:", agent_name, "FALSE:", list_bug)
                assert False
            else:
                print("Agent:", agent_name, "TRUE")
                assert True
